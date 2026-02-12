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
from src.extractor import LLMExtractor, ExtractedSpec
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
        """Initialize LLM extractor."""
        if self.extractor is None:
            provider = self.config['llm']['provider']
            
            if provider == 'ollama':
                # Use Ollama extractor
                from src.extractor import OllamaExtractor
                
                self.extractor = OllamaExtractor(
                    model=self.config['llm']['model'],
                    temperature=self.config['llm']['temperature'],
                    max_tokens=self.config['llm']['max_tokens'],
                    base_url=self.config['llm'].get('base_url', 'http://localhost:11434')
                )
                logger.debug("Ollama extractor initialized")
                
            else:
                # Use OpenAI/Anthropic extractor
                api_key = None
                if provider == 'openai':
                    api_key = os.getenv('OPENAI_API_KEY')
                elif provider == 'anthropic':
                    api_key = os.getenv('ANTHROPIC_API_KEY')
                
                from src.extractor import LLMExtractor
                
                self.extractor = LLMExtractor(
                    provider=provider,
                    model=self.config['llm']['model'],
                    temperature=self.config['llm']['temperature'],
                    max_tokens=self.config['llm']['max_tokens'],
                    api_key=api_key
                )
                logger.debug("Extractor initialized")

    
    def query(self, query_text: str, return_contexts: bool = False) -> Dict:
        """
        Query the pipeline for specifications.
        
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
                "message": "No relevant information found in the manual."
            }
        
        # Extract specifications
        specs = self.extractor.extract(
            query=query_text,
            contexts=contexts,
            validate=self.config['extraction']['validate_output']
        )

        # --- ADD THIS BLOCK ---
        if not specs and contexts:
            from src.extractor import RuleBasedExtractor
            fallback_extractor = RuleBasedExtractor()
            specs = fallback_extractor.extract(contexts)
            logger.info(f"Used rule-based extraction: {len(specs)} specs")
        
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
                    "source_chunk_id": spec.source_chunk_id
                }
                for spec in specs
            ],
            "num_results": len(specs)
        }
        
        if return_contexts:
            result["contexts"] = contexts
        
        logger.info(f"Query complete: {len(specs)} specifications extracted")
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



