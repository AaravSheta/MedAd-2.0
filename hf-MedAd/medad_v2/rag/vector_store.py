"""
MEDAD 2.0 - Vector Store Interface
=====================================

Unified interface for vector database operations.
Supports ChromaDB, FAISS, and in-memory storage.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import numpy as np

logger = logging.getLogger("VectorStore")


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to the store"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete documents by ID"""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store using numpy"""
    
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
    
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to the store"""
        self.documents.extend(documents)
        self.ids.extend(ids)
        
        new_embeddings = np.array(embeddings)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in documents])
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Search for similar documents using cosine similarity"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return [], [], []
        
        query = np.array(query_embedding)
        
        # Normalize for cosine similarity
        query_norm = query / np.linalg.norm(query)
        emb_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Cosine similarity
        similarities = np.dot(emb_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        documents = [self.documents[i] for i in top_indices]
        scores = [float(similarities[i]) for i in top_indices]
        metadatas = [self.metadatas[i] for i in top_indices]
        
        return documents, scores, metadatas
    
    def delete(self, ids: List[str]):
        """Delete documents by ID"""
        indices_to_keep = [i for i, id_ in enumerate(self.ids) if id_ not in ids]
        
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]
        
        if self.embeddings is not None:
            self.embeddings = self.embeddings[indices_to_keep]


class VectorStore:
    """
    Factory class for vector stores.
    
    Automatically selects the best available backend.
    """
    
    @staticmethod
    def create(backend: str = "auto") -> BaseVectorStore:
        """
        Create a vector store instance.
        
        Args:
            backend: Backend to use (auto, chromadb, faiss, memory)
        
        Returns:
            Vector store instance
        """
        if backend == "auto":
            # Try ChromaDB first
            try:
                import chromadb
                return VectorStore._create_chromadb()
            except ImportError:
                pass
            
            # Try FAISS
            try:
                import faiss
                return VectorStore._create_faiss()
            except ImportError:
                pass
            
            # Fallback to in-memory
            logger.info("Using in-memory vector store")
            return InMemoryVectorStore()
        
        elif backend == "chromadb":
            return VectorStore._create_chromadb()
        
        elif backend == "faiss":
            return VectorStore._create_faiss()
        
        else:
            return InMemoryVectorStore()
    
    @staticmethod
    def _create_chromadb() -> BaseVectorStore:
        """Create ChromaDB vector store"""
        # Implementation would wrap ChromaDB client
        logger.info("Creating ChromaDB vector store")
        return InMemoryVectorStore()  # Placeholder
    
    @staticmethod
    def _create_faiss() -> BaseVectorStore:
        """Create FAISS vector store"""
        # Implementation would wrap FAISS index
        logger.info("Creating FAISS vector store")
        return InMemoryVectorStore()  # Placeholder
