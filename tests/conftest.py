"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Provide a sample configuration dictionary."""
    return {
        'pdf': {
            'loader': 'pymupdf',
            'extract_images': False,
            'extract_tables': True
        },
        'text': {
            'min_chunk_size': 100,
            'max_chunk_size': 400,
            'chunk_overlap': 50,
            'remove_headers_footers': True,
            'normalize_whitespace': True
        },
        'chunking': {
            'strategy': 'semantic',
            'separator_hierarchy': ['\n\n', '\n', '. ', ' '],
            'preserve_metadata': True
        },
        'embeddings': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'batch_size': 32,
            'normalize': True,
            'device': 'cpu'
        },
        'vector_store': {
            'type': 'faiss',
            'index_type': 'IndexFlatL2',
            'persist_directory': './index',
            'index_name': 'vehicle_specs'
        },
        'retrieval': {
            'top_k': 5,
            'score_threshold': None,
            'reranking': False,
            'max_context_length': 4000
        },
        'llm': {
            'provider': 'ollama',
            'model': 'llama3',
            'base_url': 'http://localhost:11434',
            'temperature': 0.0,
            'max_tokens': 500,
            'timeout': 60
        }
    }
