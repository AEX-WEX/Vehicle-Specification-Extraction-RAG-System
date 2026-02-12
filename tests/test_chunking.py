"""
Tests for text chunking module.
"""

import pytest
from src.chunking import SemanticChunker, Chunk


class TestSemanticChunker:
    """Test semantic chunking functionality."""
    
    def test_chunker_initialization(self):
        """Test that chunker initializes with correct parameters."""
        chunker = SemanticChunker(
            min_chunk_size=200,
            max_chunk_size=800,
            chunk_overlap=100
        )
        assert chunker.min_chunk_size == 200
        assert chunker.max_chunk_size == 800
        assert chunker.chunk_overlap == 100
    
    def test_chunk_creation_basic(self):
        """Test basic chunk creation from text."""
        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=200,
            chunk_overlap=20
        )
        
        text = "This is a sample text. " * 20  # Create substantial text
        chunks = chunker.chunk_text(text, page_number=1)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(len(c.text) >= 50 for c in chunks)
    
    def test_chunk_page_numbering(self):
        """Test that chunks preserve page numbers."""
        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=200,
            chunk_overlap=20
        )
        
        text = "Sample text. " * 30
        chunks = chunker.chunk_text(text, page_number=5)
        
        assert all(c.page_number == 5 for c in chunks)
    
    def test_chunk_with_overlap(self):
        """Test that chunks maintain overlap."""
        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=150,
            chunk_overlap=30
        )
        
        text = "Word " * 100  # Create long repetitive text
        chunks = chunker.chunk_text(text, page_number=1)
        
        # Verify chunks exist and have sequential structure
        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            current_end = chunks[i].end_char
            next_start = chunks[i + 1].start_char
            # Check overlap exists
            assert next_start < current_end
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_text("", page_number=1)
        
        assert chunks == []
    
    def test_chunk_pages_multiple(self):
        """Test chunking multiple pages."""
        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=200,
            chunk_overlap=20
        )
        
        pages = [
            {"text": "Page 1 content. " * 20, "page_number": 1, "metadata": {}},
            {"text": "Page 2 content. " * 20, "page_number": 2, "metadata": {}},
        ]
        
        chunks = chunker.chunk_pages(pages)
        
        assert len(chunks) > 0
        # Verify chunk IDs are unique
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
