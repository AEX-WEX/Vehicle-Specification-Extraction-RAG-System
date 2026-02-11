"""
Retrieval Module

Handles context retrieval for RAG pipeline.
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant context from vector store.
    """
    
    def __init__(self,
                 vector_store: 'FAISSVectorStore',
                 embedding_model: 'EmbeddingModel',
                 top_k: int = 5,
                 score_threshold: Optional[float] = None,
                 max_context_length: int = 4000):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score (optional)
            max_context_length: Maximum total context length in characters
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_length = max_context_length
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant context for query.
        
        Args:
            query: User query
            
        Returns:
            List of context dictionaries
        """
        logger.info(f"Retrieving context for query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, self.top_k * 2)
        
        # Filter by score threshold if specified
        if self.score_threshold is not None:
            results = [
                (dist, chunk, meta)
                for dist, chunk, meta in results
                if self._distance_to_score(dist) >= self.score_threshold
            ]
        
        # Convert to context format
        contexts = []
        total_length = 0
        
        for dist, chunk, meta in results[:self.top_k]:
            if total_length + len(chunk.text) > self.max_context_length:
                break
            
            context = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "score": self._distance_to_score(dist),
                "distance": dist,
                "metadata": meta
            }
            
            contexts.append(context)
            total_length += len(chunk.text)
        
        logger.info(f"Retrieved {len(contexts)} context chunks")
        return contexts
    
    def _distance_to_score(self, distance: float) -> float:
        """
        Convert L2 distance to similarity score.
        
        Args:
            distance: L2 distance
            
        Returns:
            Similarity score (0-1)
        """
        # For L2 distance, convert to similarity
        # Assuming normalized embeddings, distance is in [0, 2]
        return max(0.0, 1.0 - distance / 2.0)
    
    def retrieve_with_context_window(self,
                                     query: str,
                                     window_size: int = 1) -> List[Dict]:
        """
        Retrieve with surrounding chunks for more context.
        
        Args:
            query: User query
            window_size: Number of chunks before/after to include
            
        Returns:
            List of expanded context dictionaries
        """
        # Get base contexts
        contexts = self.retrieve(query)
        
        # TODO: Implement context window expansion
        # This would require storing chunk sequences
        
        return contexts


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining semantic and keyword search.
    """
    
    def __init__(self, *args, keyword_weight: float = 0.3, **kwargs):
        """
        Initialize hybrid retriever.
        
        Args:
            keyword_weight: Weight for keyword matching (0-1)
        """
        super().__init__(*args, **kwargs)
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1.0 - keyword_weight
    
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve using hybrid approach."""
        # Get semantic results
        semantic_contexts = super().retrieve(query)
        
        # Get keyword scores
        keyword_scores = self._keyword_match(query, semantic_contexts)
        
        # Combine scores
        for ctx, kw_score in zip(semantic_contexts, keyword_scores):
            semantic_score = ctx['score']
            ctx['score'] = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * kw_score
            )
            ctx['semantic_score'] = semantic_score
            ctx['keyword_score'] = kw_score
        
        # Re-sort by combined score
        semantic_contexts.sort(key=lambda x: x['score'], reverse=True)
        
        return semantic_contexts
    
    def _keyword_match(self, query: str, contexts: List[Dict]) -> List[float]:
        """
        Calculate keyword match scores.
        
        Args:
            query: Search query
            contexts: Context dictionaries
            
        Returns:
            List of keyword scores
        """
        from collections import Counter
        
        # Tokenize query
        query_tokens = set(query.lower().split())
        
        scores = []
        for ctx in contexts:
            # Tokenize context
            ctx_tokens = ctx['text'].lower().split()
            ctx_counter = Counter(ctx_tokens)
            
            # Calculate overlap
            matches = sum(ctx_counter[token] for token in query_tokens)
            score = matches / max(len(query_tokens), 1)
            
            scores.append(min(score, 1.0))
        
        return scores


if __name__ == "__main__":
    print("Retriever module - use within pipeline")
