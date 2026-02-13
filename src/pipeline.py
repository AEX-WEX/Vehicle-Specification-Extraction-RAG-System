"""
RAG Pipeline Module

Orchestrates the complete extraction pipeline.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional
import os

from src.pdf_loader import load_pdf
from src.chunking import create_chunker, Chunk
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.extractor import ExtractedSpec, OllamaExtractor
from src.utils import (
    get_pdf_hash,
    load_indexed_pdf_metadata,
    save_indexed_pdf_metadata,
    is_pdf_different,
)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class VehicleSpecRAGPipeline:
    """
    Complete RAG pipeline for vehicle specification extraction.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline from config.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.extractor = None
        
        logger.info("Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def build_index(self, pdf_path: str, force_rebuild: bool = False):
        """
        Build vector index from PDF.

        Args:
            pdf_path: Path to service manual PDF
            force_rebuild: Force rebuild even if index exists
        """
        logger.info(f"Building index from PDF: {pdf_path}")

        # Check if index already exists and if PDF has changed
        index_dir = self.config['vector_store']['persist_directory']
        index_name = self.config['vector_store']['index_name']
        index_path = Path(index_dir) / f"{index_name}.faiss"

        # Determine if we need to rebuild
        need_rebuild = force_rebuild

        if not need_rebuild and index_path.exists():
            # Check if PDF is different from previous indexed PDF
            if is_pdf_different(pdf_path, index_dir):
                logger.info("Different PDF detected - will rebuild index")
                need_rebuild = True
            else:
                logger.info("Index already exists and PDF matches - using existing index")
                self.load_index()
                return

        if not need_rebuild and not index_path.exists():
            need_rebuild = True

        if not need_rebuild:
            logger.info("Index already exists. Use force_rebuild=True to rebuild.")
            self.load_index()
            return

        # Step 1: Extract text from PDF
        logger.info("Step 1/5: Extracting text from PDF...")
        pages = load_pdf(
            pdf_path=pdf_path,
            loader_type=self.config['pdf']['loader'],
            clean=True,
            remove_headers_footers=self.config['text']['remove_headers_footers'],
            normalize_whitespace=self.config['text']['normalize_whitespace']
        )
        logger.info(f"Extracted {len(pages)} pages")

        # Step 2: Chunk text
        logger.info("Step 2/5: Chunking text...")
        chunker = create_chunker(
            strategy=self.config['chunking']['strategy'],
            min_chunk_size=self.config['text']['min_chunk_size'],
            max_chunk_size=self.config['text']['max_chunk_size'],
            chunk_overlap=self.config['text']['chunk_overlap']
        )
        chunks = chunker.chunk_pages(pages)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Initialize embedding model
        logger.info("Step 3/5: Initializing embedding model...")
        self.embedding_model = EmbeddingModel(
            model_name=self.config['embeddings']['model_name'],
            device=self.config['embeddings']['device'],
            batch_size=self.config['embeddings']['batch_size'],
            normalize=self.config['embeddings']['normalize']
        )

        # Step 4: Generate embeddings
        logger.info("Step 4/5: Generating embeddings...")
        embeddings = self.embedding_model.embed_chunks(chunks)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")

        # Step 5: Build vector store
        logger.info("Step 5/5: Building vector index...")
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_model.get_embedding_dim(),
            index_type=self.config['vector_store']['index_type'],
            persist_directory=index_dir
        )
        self.vector_store.add_embeddings(embeddings, chunks)

        # Save index
        os.makedirs(index_dir, exist_ok=True)
        self.vector_store.save(index_name)
        logger.info(f"Index saved to {index_dir}")

        # Save metadata about the indexed PDF
        save_indexed_pdf_metadata(
            index_dir=index_dir,
            pdf_path=pdf_path,
            num_chunks=len(chunks),
            embedding_model=self.config['embeddings']['model_name'],
            index_type=self.config['vector_store']['index_type']
        )

        # Initialize retriever
        self._init_retriever()

        logger.info("Index build complete")
    
    def load_index(self):
        """Load existing vector index."""
        logger.info("Loading existing index...")
        
        # Initialize embedding model
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel(
                model_name=self.config['embeddings']['model_name'],
                device=self.config['embeddings']['device'],
                batch_size=self.config['embeddings']['batch_size'],
                normalize=self.config['embeddings']['normalize']
            )
        
        # Load vector store
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_model.get_embedding_dim(),
            persist_directory=self.config['vector_store']['persist_directory']
        )
        self.vector_store.load(self.config['vector_store']['index_name'])
        
        # Initialize retriever
        self._init_retriever()
        
        stats = self.vector_store.get_stats()
        logger.info(f"Index loaded: {stats['total_chunks']} chunks")
    
    def _init_retriever(self):
        """Initialize retriever."""
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            top_k=self.config['retrieval']['top_k'],
            score_threshold=self.config['retrieval'].get('score_threshold'),
            max_context_length=self.config['retrieval']['max_context_length']
        )
        logger.debug("Retriever initialized")
    
    def _init_extractor(self):
        """Initialize SmartExtractor (compares Ollama vs rule-based)."""
        if self.extractor is None:
            provider = self.config['llm']['provider']
            
            if provider == 'ollama':
                # Use SmartExtractor which compares Ollama and rule-based
                from src.extractor import SmartExtractor
                
                self.extractor = SmartExtractor()
                logger.debug("SmartExtractor initialized (will compare Ollama vs rule-based)")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}. Only 'ollama' is supported.")

    
    def query(self, query_text: str, return_contexts: bool = False) -> Dict:
        """
        Query the pipeline for specifications.
        
        SmartExtractor will:
        - Try both Ollama and rule-based extraction
        - Compare results
        - Return the BEST results
        - Track which method was used
        
        Args:
            query_text: User query
            return_contexts: Whether to return retrieved contexts
            
        Returns:
            Dictionary with extracted specifications and metadata
        """
        logger.info(f"Processing query: {query_text}")
        
        # Ensure index is loaded
        if self.vector_store is None:
            logger.info("Loading index...")
            self.load_index()
        
        # Ensure extractor is initialized
        self._init_extractor()
        
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve(query_text)
        
        if not contexts:
            logger.warning("No relevant contexts found")
            return {
                "query": query_text,
                "specifications": [],
                "extraction_method": None,
                "average_confidence": 0.0,
                "message": "No relevant information found in the manual."
            }
        
        # Extract specifications with method tracking
        metadata = self.extractor.extract_with_metadata(
            query=query_text,
            contexts=contexts
        )
        
        specs = metadata["specs"]
        method_used = metadata["method_used"]
        avg_confidence = metadata["average_confidence"]
        
        logger.info(f"Extraction complete: {len(specs)} specs via {method_used} (confidence: {avg_confidence:.2f})")
        
        # Format results
        result = {
            "query": query_text,
            "specifications": [
                {
                    "component": spec.component,
                    "spec_type": spec.spec_type,
                    "value": spec.value,
                    "unit": spec.unit,
                    "page_number": spec.page_number,
                    "source_chunk_id": spec.source_chunk_id,
                    "confidence": spec.confidence
                }
                for spec in specs
            ],
            "num_results": len(specs),
            "extraction_method": method_used,
            "average_confidence": avg_confidence,
            "message": metadata["message"]
        }
        
        if return_contexts:
            result["contexts"] = contexts
        
        return result
    
    def get_status(self) -> Dict:
        """Get pipeline status."""
        status = {
            "initialized": True,
            "embedding_model_loaded": self.embedding_model is not None,
            "index_loaded": self.vector_store is not None,
            "extractor_initialized": self.extractor is not None
        }
        
        if self.vector_store is not None:
            status.update(self.vector_store.get_stats())
        
        return status


def create_pipeline(config_path: str = "config.yaml") -> VehicleSpecRAGPipeline:
    """
    Factory function to create pipeline.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Pipeline instance
    """
    return VehicleSpecRAGPipeline(config_path)



