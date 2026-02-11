"""
Vector Store Module

Handles vector storage and similarity search using FAISS.
"""

import os
import pickle
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search.
    """
    
    def __init__(self,
                 embedding_dim: int,
                 index_type: str = "IndexFlatL2",
                 persist_directory: Optional[str] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type
            persist_directory: Directory to persist index
        """
        import faiss
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.persist_directory = persist_directory
        
        # Create index
        if index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Storage for chunks and metadata
        self.chunks = []
        self.metadata = []
        
        logger.info(f"Initialized FAISS index: {index_type} (dim={embedding_dim})")
    
    def add_embeddings(self,
                       embeddings: np.ndarray,
                       chunks: List['Chunk']):
        """
        Add embeddings and chunks to the index.
        
        Args:
            embeddings: Embedding vectors (n_chunks, embedding_dim)
            chunks: List of Chunk objects
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # Train index if needed (for IVF)
        if self.index_type.startswith("IndexIVF") and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        self.metadata.extend([{
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            **chunk.metadata
        } for chunk in chunks])
        
        logger.info(f"Added {len(chunks)} chunks to index. Total: {len(self.chunks)}")
    
    def search(self,
               query_embedding: np.ndarray,
               top_k: int = 5) -> List[Tuple[float, 'Chunk', Dict]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (distance, chunk, metadata) tuples
        """
        if len(self.chunks) == 0:
            logger.warning("Index is empty")
            return []
        
        # Ensure correct shape and type
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((
                    float(dist),
                    self.chunks[idx],
                    self.metadata[idx]
                ))
        
        logger.debug(f"Found {len(results)} results for query")
        return results
    
    def save(self, index_name: str = "vehicle_specs"):
        """
        Save index and metadata to disk.
        
        Args:
            index_name: Name for the saved index
        """
        import faiss
        
        if self.persist_directory is None:
            logger.warning("No persist directory specified")
            return
        
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(self.persist_directory, f"{index_name}.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save chunks and metadata
        data_path = os.path.join(self.persist_directory, f"{index_name}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved index to {self.persist_directory}")
    
    def load(self, index_name: str = "vehicle_specs"):
        """
        Load index and metadata from disk.
        
        Args:
            index_name: Name of the saved index
        """
        import faiss
        
        if self.persist_directory is None:
            raise ValueError("No persist directory specified")
        
        # Load FAISS index
        index_path = os.path.join(self.persist_directory, f"{index_name}.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load chunks and metadata
        data_path = os.path.join(self.persist_directory, f"{index_name}.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data['index_type']
        
        logger.info(f"Loaded index from {self.persist_directory} ({len(self.chunks)} chunks)")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }


if __name__ == "__main__":
    # Test vector store
    from src.chunking import Chunk
    
    # Create sample data
    embedding_dim = 384
    n_chunks = 100
    
    embeddings = np.random.randn(n_chunks, embedding_dim).astype('float32')
    chunks = [
        Chunk(
            text=f"Sample chunk {i}",
            chunk_id=f"chunk_{i:05d}",
            page_number=1,
            start_char=0,
            end_char=100,
            metadata={}
        )
        for i in range(n_chunks)
    ]
    
    # Create and populate store
    store = FAISSVectorStore(embedding_dim, persist_directory="./test_index")
    store.add_embeddings(embeddings, chunks)
    
    # Search
    query_emb = np.random.randn(embedding_dim).astype('float32')
    results = store.search(query_emb, top_k=5)
    
    print(f"Found {len(results)} results")
    for dist, chunk, meta in results:
        print(f"  Distance: {dist:.3f}, Chunk: {chunk.chunk_id}")
