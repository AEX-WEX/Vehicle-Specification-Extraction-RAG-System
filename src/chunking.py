"""
Text Chunking Module

Implements semantic chunking strategies for optimal retrieval performance.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    page_number: int
    start_char: int
    end_char: int
    metadata: Dict[str, any]


class SemanticChunker:
    """
    Semantic-aware text chunker.
    
    Splits text at natural boundaries while respecting size constraints.
    """
    
    def __init__(self,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 separator_hierarchy: Optional[List[str]] = None):
        """
        Initialize semantic chunker.
        
        Args:
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            separator_hierarchy: List of separators in priority order
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separator_hierarchy is None:
            self.separator_hierarchy = ["\n\n", "\n", ". ", " "]
        else:
            self.separator_hierarchy = separator_hierarchy
    
    def chunk_pages(self, pages: List[Dict[str, any]]) -> List[Chunk]:
        """
        Chunk multiple pages.
        
        Args:
            pages: List of page dictionaries from PDF loader
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        chunk_counter = 0
        
        for page in pages:
            page_chunks = self.chunk_text(
                text=page["text"],
                page_number=page["page_number"],
                metadata=page.get("metadata", {})
            )
            
            for chunk in page_chunks:
                chunk.chunk_id = f"chunk_{chunk_counter:05d}"
                chunk_counter += 1
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def chunk_text(self, text: str, page_number: int, metadata=None) -> List[Chunk]:
        """
        Split text into chunks with optional metadata.
        
        Args:
            text: Text to chunk
            page_number: Source page number
            metadata: Additional metadata
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        chunks = []
        step = self.max_chunk_size - self.chunk_overlap
        text_length = len(text)

        for start_pos in range(0, text_length, step):
            end_pos = min(start_pos + self.max_chunk_size, text_length)

            chunk_text = text[start_pos:end_pos].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id="",
                        page_number=page_number,
                        start_char=start_pos,
                        end_char=end_pos,
                        metadata=metadata or {},
                    )
                )

        return chunks
    
    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """
        Find optimal split point using separator hierarchy.
        
        Args:
            text: Full text
            start: Chunk start position
            end: Target end position
            
        Returns:
            Adjusted end position at semantic boundary
        """
        # Search window around target end position
        search_start = max(start + self.min_chunk_size, end - 200)
        search_end = min(end + 100, len(text))
        search_text = text[search_start:search_end]
        
        # Try each separator in hierarchy
        for separator in self.separator_hierarchy:
            # Find last occurrence of separator
            last_idx = search_text.rfind(separator)
            
            if last_idx != -1:
                # Calculate absolute position
                split_pos = search_start + last_idx + len(separator)
                
                # Validate chunk size
                if split_pos - start >= self.min_chunk_size:
                    return split_pos
        
        # Fallback to target position
        return end


class RecursiveChunker:
    """
    Recursive text chunker that splits progressively.
    
    Useful for very structured documents.
    """
    
    def __init__(self,
                 chunk_size: int = 600,
                 chunk_overlap: int = 100):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
        
    def chunk_pages(self, pages: List[Dict[str, any]]) -> List[Chunk]:
        """Chunk pages recursively."""
        all_chunks = []
        chunk_counter = 0
        
        for page in pages:
            splits = self._recursive_split(page["text"], self.separators)
            
            for split_text in splits:
                chunk = Chunk(
                    text=split_text,
                    chunk_id=f"chunk_{chunk_counter:05d}",
                    page_number=page["page_number"],
                    start_char=0,
                    end_char=len(split_text),
                    metadata=page.get("metadata", {})
                )
                chunk_counter += 1
                all_chunks.append(chunk)
        
        return all_chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator) if separator else [text]
        
        chunks = []
        for split in splits:
            if len(split) <= self.chunk_size:
                if split.strip():
                    chunks.append(split)
            else:
                chunks.extend(self._recursive_split(split, remaining_separators))
        
        return chunks


def create_chunker(strategy: str = "semantic", **kwargs) -> any:
    """
    Factory function to create chunker.
    
    Args:
        strategy: Chunking strategy ("semantic" or "recursive")
        **kwargs: Strategy-specific parameters
        
    Returns:
        Chunker instance
    """
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    elif strategy == "recursive":
        return RecursiveChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")



