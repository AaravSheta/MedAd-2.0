"""
MEDAD 2.0 - Embedding Manager
================================

Manages embedding creation, caching, and retrieval for the semantic search engine.
Supports multiple embedding models and efficient batch processing.
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import os
import pickle
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("EmbeddingManager")


class EmbeddingManager:
    """
    Manages embeddings for the semantic search engine.
    
    Features:
    - Multi-model support (BioBERT, ClinicalBERT, MiniLM)
    - Efficient batch processing
    - Disk caching for persistence
    - Incremental updates
    """
    
    def __init__(self, config, cache_dir: str = ".cache/embeddings"):
        """
        Initialize the embedding manager.
        
        Args:
            config: SemanticEngineConfig
            cache_dir: Directory for caching embeddings
        """
        self.config = config
        self.cache_dir = cache_dir
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate cache key from texts and model"""
        content = f"{model_name}:{','.join(sorted(texts[:100]))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def load_model(self, model_name: str) -> bool:
        """Load the embedding model"""
        loop = asyncio.get_event_loop()
        
        def _load():
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                return True
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return False
        
        return await loop.run_in_executor(self._executor, _load)
    
    async def create_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            use_cache: Use disk cache if available
        
        Returns:
            Numpy array of embeddings
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(texts, str(self.config.primary_model))
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                logger.info(f"Loading embeddings from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Create embeddings
        loop = asyncio.get_event_loop()
        
        def _encode():
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
        
        embeddings = await loop.run_in_executor(self._executor, _encode)
        
        # Cache embeddings
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Cached embeddings to: {cache_path}")
        
        return embeddings
    
    async def update_embeddings(
        self,
        existing_embeddings: np.ndarray,
        new_texts: List[str],
        indices: List[int]
    ) -> np.ndarray:
        """
        Update specific embeddings in an existing array.
        
        Args:
            existing_embeddings: Existing embedding array
            new_texts: New texts to embed
            indices: Indices to update
        
        Returns:
            Updated embedding array
        """
        new_embeddings = await self.create_embeddings(new_texts, use_cache=False)
        
        result = existing_embeddings.copy()
        for i, idx in enumerate(indices):
            result[idx] = new_embeddings[i]
        
        return result
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return self.config.embedding_dim
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)
        self._model = None
