"""
MEDAD 2.0 - Knowledge Retrieval Module (RAG)
===============================================

Retrieval-Augmented Generation for enhanced medical responses.

Key Features:
- Vector database for medical knowledge storage
- Semantic retrieval of relevant documents
- Re-ranking for improved precision
- Context injection into LLM prompts
- Source attribution for transparency

Knowledge Sources:
- Medicine database (248K+ entries)
- Drug interaction databases
- Clinical guidelines
- Symptom-disease mappings
- Treatment protocols

The RAG pipeline enhances Gemini responses by providing
relevant medical context from our knowledge base.
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("KnowledgeRetriever")


@dataclass
class RetrievalResult:
    """Result of knowledge retrieval"""
    query: str
    context: str
    sources: List[Dict[str, str]]
    relevance_scores: List[float]
    retrieval_time_ms: float


@dataclass
class Document:
    """A document in the knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class KnowledgeRetriever:
    """
    RAG-based knowledge retrieval for medical context.
    
    Uses vector similarity search to find relevant medical knowledge
    and inject it into LLM prompts for more accurate responses.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    User Query                                │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │              Query Embedding                                 │
    │         (Sentence Transformer)                              │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │           Vector Similarity Search                          │
    │    (ChromaDB / FAISS / Pinecone)                           │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │              Re-ranking                                      │
    │    (Cross-Encoder for precision)                            │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │           Context Aggregation                                │
    │    (Combine top-k documents)                                │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │           LLM Prompt Injection                              │
    │    (Augment Gemini prompt with context)                    │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config=None):
        """
        Initialize the knowledge retriever.
        
        Args:
            config: RAGConfig with retrieval parameters (optional)
        """
        # Create default config if not provided
        if config is None:
            from ..core.config import RAGConfig
            config = RAGConfig()
        
        self.config = config
        self._vector_store = None
        self._embedding_model = None
        self._reranker = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False
        
        logger.info("KnowledgeRetriever initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the RAG components.
        
        1. Load embedding model
        2. Initialize vector store
        3. Load reranker model (optional)
        """
        loop = asyncio.get_event_loop()
        
        def _init():
            try:
                # Initialize embedding model
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded")
                
                # Initialize vector store
                try:
                    import chromadb
                    from chromadb.config import Settings
                    
                    self._vector_store = chromadb.Client(Settings(
                        anonymized_telemetry=False
                    ))
                    
                    # Create or get collection
                    self._collection = self._vector_store.get_or_create_collection(
                        name=self.config.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info("✅ ChromaDB initialized")
                except ImportError:
                    logger.warning("ChromaDB not available, using in-memory store")
                    self._vector_store = None
                
                # Initialize reranker (optional)
                if self.config.rerank_enabled:
                    try:
                        from sentence_transformers import CrossEncoder
                        self._reranker = CrossEncoder(self.config.rerank_model)
                        logger.info("✅ Reranker loaded")
                    except Exception as e:
                        logger.warning(f"Reranker not available: {e}")
                
                return True
                
            except Exception as e:
                logger.error(f"RAG initialization failed: {e}")
                return False
        
        success = await loop.run_in_executor(self._executor, _init)
        self._initialized = success
        return success
    
    async def index_medicines(self, medicines_df):
        """
        Index medicine data into the vector store.
        
        Args:
            medicines_df: Pandas DataFrame with medicine data
        """
        if not self._initialized:
            logger.warning("RAG not initialized, cannot index")
            return
        
        loop = asyncio.get_event_loop()
        
        def _index():
            documents = []
            embeddings = []
            ids = []
            metadatas = []
            
            # Process medicines in batches
            batch_size = 1000
            total = len(medicines_df)
            
            for i in range(0, total, batch_size):
                batch = medicines_df.iloc[i:i+batch_size]
                
                for idx, row in batch.iterrows():
                    # Create document content
                    content = f"""
                    Medicine: {row.get('name', row.get('Medicine Name', 'Unknown'))}
                    Therapeutic Class: {row.get('Therapeutic Class', 'General')}
                    Uses: {row.get('combined_use', '')}
                    Manufacturer: {row.get('Manufacturer', 'N/A')}
                    """
                    
                    doc_id = f"med_{idx}"
                    
                    documents.append(content)
                    ids.append(doc_id)
                    metadatas.append({
                        "name": str(row.get('name', row.get('Medicine Name', ''))),
                        "therapeutic_class": str(row.get('Therapeutic Class', '')),
                        "type": "medicine"
                    })
                
                logger.info(f"Processing {i+len(batch)}/{total} medicines...")
            
            # Create embeddings
            embeddings = self._embedding_model.encode(documents, show_progress_bar=True)
            
            # Add to vector store
            if self._collection is not None:
                # ChromaDB
                self._collection.add(
                    documents=documents,
                    embeddings=embeddings.tolist(),
                    ids=ids,
                    metadatas=metadatas
                )
            else:
                # In-memory fallback
                self._documents = documents
                self._embeddings = embeddings
                self._ids = ids
                self._metadatas = metadatas
            
            logger.info(f"✅ Indexed {len(documents)} medicines")
        
        await loop.run_in_executor(self._executor, _index)
    
    async def retrieve(
        self,
        query: str,
        medicines: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> RetrievalResult:
        """
        Retrieve relevant medical knowledge for a query.
        
        Args:
            query: User's symptom query
            medicines: Optional list of already-found medicines for context
            top_k: Number of documents to retrieve
        
        Returns:
            RetrievalResult with context and sources
        """
        import time
        start_time = time.time()
        
        if not self._initialized:
            return RetrievalResult(
                query=query,
                context="",
                sources=[],
                relevance_scores=[],
                retrieval_time_ms=0.0
            )
        
        loop = asyncio.get_event_loop()
        
        def _retrieve():
            # Encode query
            query_embedding = self._embedding_model.encode([query])[0]
            
            # Search vector store
            if self._collection is not None:
                results = self._collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k * 2  # Get more for reranking
                )
                
                documents = results['documents'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0]
            else:
                # In-memory fallback
                if not hasattr(self, '_embeddings') or self._embeddings is None:
                    return [], [], []
                
                import numpy as np
                similarities = np.dot(self._embeddings, query_embedding)
                top_indices = np.argsort(similarities)[-top_k*2:][::-1]
                
                documents = [self._documents[i] for i in top_indices]
                distances = [1 - similarities[i] for i in top_indices]
                metadatas = [self._metadatas[i] for i in top_indices]
            
            return documents, distances, metadatas
        
        documents, distances, metadatas = await loop.run_in_executor(
            self._executor, _retrieve
        )
        
        if not documents:
            return RetrievalResult(
                query=query,
                context="",
                sources=[],
                relevance_scores=[],
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
        
        # Optional: Rerank results
        if self._reranker and len(documents) > top_k:
            def _rerank():
                pairs = [[query, doc] for doc in documents]
                scores = self._reranker.predict(pairs)
                
                # Sort by rerank score
                sorted_results = sorted(
                    zip(documents, distances, metadatas, scores),
                    key=lambda x: x[3],
                    reverse=True
                )
                
                return [list(x) for x in zip(*sorted_results[:top_k])]
            
            documents, distances, metadatas, rerank_scores = await loop.run_in_executor(
                self._executor, _rerank
            )
            relevance_scores = rerank_scores
        else:
            # Use original distances as scores (convert to similarity)
            documents = documents[:top_k]
            distances = distances[:top_k]
            metadatas = metadatas[:top_k]
            relevance_scores = [1 - d for d in distances]
        
        # Aggregate context
        context = self._aggregate_context(documents, metadatas, medicines)
        
        # Create source attribution
        sources = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            sources.append({
                "type": meta.get("type", "document"),
                "name": meta.get("name", f"Source {i+1}"),
                "therapeutic_class": meta.get("therapeutic_class", ""),
                "relevance": relevance_scores[i] if i < len(relevance_scores) else 0.0
            })
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            context=context,
            sources=sources,
            relevance_scores=relevance_scores,
            retrieval_time_ms=retrieval_time
        )
    
    def _aggregate_context(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        medicines: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Aggregate retrieved documents into a context string.
        
        Formats the context for optimal LLM consumption.
        """
        context_parts = []
        
        # Add retrieved documents
        if documents:
            context_parts.append("=== Relevant Medical Knowledge ===")
            for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
                context_parts.append(f"\n[{i}] {meta.get('name', 'Source')}")
                context_parts.append(doc.strip())
        
        # Add medicine context if provided
        if medicines:
            context_parts.append("\n=== Related Medicines ===")
            for med in medicines[:3]:
                context_parts.append(
                    f"- {med.get('Medicine Name', 'Unknown')}: "
                    f"{med.get('Uses', 'General use')[:100]}"
                )
        
        return "\n".join(context_parts)
    
    async def add_knowledge(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ):
        """
        Add new knowledge to the vector store.
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID
        """
        if not self._initialized:
            return
        
        loop = asyncio.get_event_loop()
        
        def _add():
            import uuid
            
            doc_id_to_use = doc_id or str(uuid.uuid4())
            embedding = self._embedding_model.encode([content])[0]
            
            if self._collection is not None:
                self._collection.add(
                    documents=[content],
                    embeddings=[embedding.tolist()],
                    ids=[doc_id_to_use],
                    metadatas=[metadata]
                )
            else:
                # In-memory fallback
                if not hasattr(self, '_documents'):
                    self._documents = []
                    self._embeddings = []
                    self._ids = []
                    self._metadatas = []
                
                import numpy as np
                self._documents.append(content)
                self._embeddings = np.vstack([self._embeddings, embedding]) if len(self._embeddings) > 0 else np.array([embedding])
                self._ids.append(doc_id_to_use)
                self._metadatas.append(metadata)
        
        await loop.run_in_executor(self._executor, _add)
    
    async def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)
        self._embedding_model = None
        self._reranker = None
        self._initialized = False
        logger.info("KnowledgeRetriever cleaned up")
