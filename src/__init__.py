"""
Vehicle Specification Extraction RAG System

A production-quality pipeline for extracting structured specifications
from automotive service manual PDFs using retrieval-augmented generation.
"""

__version__ = "1.0.0"
__author__ = "Aditya Kumar Choudhary"

# Public API, imported lazily so that `import src.table_parser` does not load torch/faiss.
_EXPORTS = {
    "VehicleSpecRAGPipeline": "src.pipeline", "create_pipeline": "src.pipeline",
    "get_pdf_hash": "src.utils", "load_indexed_pdf_metadata": "src.utils",
    "save_indexed_pdf_metadata": "src.utils", "is_pdf_different": "src.utils",
    "load_pdf": "src.pdf_loader", "PyMuPDFLoader": "src.pdf_loader", "PDFMinerLoader": "src.pdf_loader",
    "TextCleaner": "src.pdf_loader",
    "Chunk": "src.chunking", "SemanticChunker": "src.chunking", "RecursiveChunker": "src.chunking",
    "create_chunker": "src.chunking",
    "EmbeddingModel": "src.embeddings", "CachedEmbeddingModel": "src.embeddings",
    "FAISSVectorStore": "src.vector_store",
    "Retriever": "src.retriever", "HybridRetriever": "src.retriever",
    "ExtractedSpec": "src.extractor", "RuleBasedExtractor": "src.extractor", "OllamaExtractor": "src.extractor",
    "SmartExtractor": "src.extractor", "TableRowExtractor": "src.extractor",
    "SpecificationEvaluator": "src.evaluation", "RetrievalEvaluator": "src.evaluation",
    "GroundTruthSpec": "src.evaluation", "EvaluationResult": "src.evaluation", "load_ground_truth": "src.evaluation",
}


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
