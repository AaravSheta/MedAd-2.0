"""
MEDAD 2.0 - Multimodal Medical AI Assistant
=============================================

A next-generation clinical assistant integrating:
- Transformer-based semantic understanding (BioBERT/ClinicalBERT)
- Computer vision for dermatological assessment
- Hinglish/Romanized Hindi NLP processing
- 3D medical visualization
- Retrieval-Augmented Generation (RAG)
- Voice-activated interfaces

Architecture:
├── semantic_engine/     - Transformer-based symptom understanding
├── vision/              - Dermatological image analysis
├── nlp/                 - Hinglish processing & transliteration
├── voice/               - Speech-to-text interfaces
├── visualization/       - 3D anatomy & drug interaction graphs
├── rag/                 - Vector database & knowledge retrieval
└── core/                - Main orchestration & API

Author: MedAd Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "MedAd Team"

from .core.orchestrator import MedAdOrchestrator
from .semantic_engine.transformer_search import TransformerSearchEngine
from .nlp.hinglish_processor import HinglishProcessor
from .vision.derma_analyzer import DermaAnalyzer
from .rag.knowledge_retriever import KnowledgeRetriever
