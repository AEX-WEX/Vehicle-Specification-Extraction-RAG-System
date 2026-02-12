"""
Vehicle Specification Extraction RAG System

A production-quality pipeline for extracting structured specifications
from automotive service manual PDFs using retrieval-augmented generation.
"""

__version__ = "1.0.0"
__author__ = "Vehicle Spec RAG Team"

# Expose main pipeline interface
from src.pipeline import VehicleSpecRAGPipeline, create_pipeline

# Expose utility functions
from src.utils import (
    get_pdf_hash,
    load_indexed_pdf_metadata,
    save_indexed_pdf_metadata,
    is_pdf_different,
)

# Expose core components for advanced usage
from src.pdf_loader import load_pdf, PyMuPDFLoader, PDFMinerLoader, TextCleaner
from src.chunking import Chunk, SemanticChunker, RecursiveChunker, create_chunker
from src.embeddings import EmbeddingModel, CachedEmbeddingModel
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever, HybridRetriever
from src.extractor import LLMExtractor, ExtractedSpec, RuleBasedExtractor
from src.evaluation import (
    SpecificationEvaluator,
    RetrievalEvaluator,
    GroundTruthSpec,
    EvaluationResult,
    load_ground_truth
)

__all__ = [
    # Main pipeline
    "VehicleSpecRAGPipeline",
    "create_pipeline",
    
    # Utilities
    "get_pdf_hash",
    "load_indexed_pdf_metadata",
    "save_indexed_pdf_metadata",
    "is_pdf_different",
    
    # PDF processing
    "load_pdf",
    "PyMuPDFLoader",
    "PDFMinerLoader",
    "TextCleaner",
    
    # Chunking
    "Chunk",
    "SemanticChunker",
    "RecursiveChunker",
    "create_chunker",
    
    # Embeddings
    "EmbeddingModel",
    "CachedEmbeddingModel",
    
    # Vector storage
    "FAISSVectorStore",
    
    # Retrieval
    "Retriever",
    "HybridRetriever",
    
    # Extraction
    "LLMExtractor",
    "ExtractedSpec",
    "RuleBasedExtractor",
    
    # Evaluation
    "SpecificationEvaluator",
    "RetrievalEvaluator",
    "GroundTruthSpec",
    "EvaluationResult",
    "load_ground_truth",
]
