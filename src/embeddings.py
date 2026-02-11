"""
Embeddings Module

Handles text embedding generation using sentence transformers.
"""

import logging
import numpy as np
from typing import List, Optional, Union
import torch

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence transformer embedding models.
    """
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 normalize: bool = True):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cpu", "cuda", or "mps")
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        """
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self,
                    texts: List[str],
                    show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.debug(f"Embedding {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embedding[0]
    
    def embed_chunks(self, chunks: List['Chunk']) -> np.ndarray:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, show_progress=True)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def similarity(self,
                   embedding1: np.ndarray,
                   embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        return cosine_similarity(embedding1, embedding2)[0][0]


class CachedEmbeddingModel:
    """
    Embedding model with caching for repeated queries.
    """
    
    def __init__(self,
                 base_model: EmbeddingModel,
                 cache_size: int = 1000):
        """
        Initialize cached embedding model.
        
        Args:
            base_model: Base embedding model
            cache_size: Maximum cache size
        """
        self.base_model = base_model
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with caching."""
        if query in self._cache:
            return self._cache[query]
        
        embedding = self.base_model.embed_query(query)
        
        # Add to cache
        self._cache[query] = embedding
        self._cache_order.append(query)
        
        # Evict oldest if cache full
        if len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        return embedding
    
    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Delegate to base model."""
        return self.base_model.embed_texts(texts, show_progress)
    
    def embed_chunks(self, chunks: List['Chunk']) -> np.ndarray:
        """Delegate to base model."""
        return self.base_model.embed_chunks(chunks)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.base_model.get_embedding_dim()


if __name__ == "__main__":
    # Test embeddings
    model = EmbeddingModel()
    
    texts = [
        "Brake caliper bolt torque is 35 Nm",
        "Engine oil capacity is 4.5 liters",
        "Tire pressure should be 2.5 bar"
    ]
    
    embeddings = model.embed_texts(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    query = "What is the brake caliper torque?"
    query_emb = model.embed_query(query)
    print(f"Query embedding shape: {query_emb.shape}")
    
    # Calculate similarities
    for i, text in enumerate(texts):
        sim = model.similarity(query_emb, embeddings[i])
        print(f"Similarity with '{text[:50]}...': {sim:.3f}")
