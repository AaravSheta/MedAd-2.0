"""
MEDAD 2.0 - Configuration Management
=======================================

Centralized configuration for all MedAd 2.0 modules.
Supports environment variables and dynamic configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelSize(Enum):
    """Model size options for resource-constrained environments"""
    NANO = "nano"       # ~50MB, fastest, lowest accuracy
    SMALL = "small"     # ~100MB, balanced
    MEDIUM = "medium"   # ~500MB, good accuracy
    LARGE = "large"     # ~1GB+, best accuracy
    

class SearchMode(Enum):
    """Search algorithm modes"""
    TFIDF_ONLY = "tfidf"           # Legacy TF-IDF only
    SEMANTIC_ONLY = "semantic"      # Transformer embeddings only
    HYBRID = "hybrid"               # Combined TF-IDF + Semantic
    RAG = "rag"                     # RAG with knowledge retrieval


@dataclass
class SemanticEngineConfig:
    """Configuration for transformer-based semantic search"""
    # Model selection - prioritize biomedical models
    primary_model: str = "dmis-lab/biobert-base-cased-v1.2"
    fallback_model: str = "all-MiniLM-L6-v2"  # Lightweight fallback
    clinical_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Embedding parameters
    embedding_dim: int = 768
    max_sequence_length: int = 512
    batch_size: int = 32
    
    # Search parameters
    similarity_threshold: float = 0.35
    top_k_results: int = 20
    use_gpu: bool = True
    
    # Cache settings
    cache_embeddings: bool = True
    cache_dir: str = ".cache/embeddings"
    
    # Index settings (for FAISS/Annoy)
    index_type: str = "faiss"  # faiss, annoy, or simple
    nprobe: int = 10  # FAISS IVF nprobe


@dataclass 
class HinglishConfig:
    """Configuration for Hinglish/Romanized Hindi NLP"""
    # Phonetic matching
    enable_phonetic_matching: bool = True
    phonetic_algorithm: str = "soundex_hindi"  # soundex_hindi, metaphone, fuzzy
    
    # Transliteration
    enable_transliteration: bool = True
    transliteration_model: str = "ai4bharat/IndicTrans2"
    
    # Common Hinglish medical mappings (sar dard -> headache)
    custom_mappings_file: str = "data/hinglish_medical_terms.json"
    
    # Fuzzy matching threshold
    fuzzy_threshold: float = 0.7
    
    # Language detection
    detect_language: bool = True
    supported_languages: List[str] = field(default_factory=lambda: ["en", "hi", "hinglish"])


@dataclass
class VisionConfig:
    """Configuration for computer vision / dermatological analysis"""
    # Model selection
    dermnet_model: str = "microsoft/resnet-50"  # Base vision model
    skin_classifier: str = "dermnet"            # dermnet, isic, custom
    
    # Image processing
    image_size: tuple = (224, 224)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Detection parameters
    confidence_threshold: float = 0.6
    top_k_conditions: int = 5
    
    # Supported conditions
    supported_conditions: List[str] = field(default_factory=lambda: [
        "acne", "eczema", "psoriasis", "melanoma", "rosacea",
        "ringworm", "vitiligo", "urticaria", "dermatitis", "herpes"
    ])
    
    # Safety
    enable_content_filter: bool = True
    require_disclaimer: bool = True


@dataclass
class VoiceConfig:
    """Configuration for voice input/output"""
    # Speech-to-Text
    stt_engine: str = "whisper"  # whisper, google, azure
    whisper_model: str = "base"  # tiny, base, small, medium, large
    
    # Supported languages
    languages: List[str] = field(default_factory=lambda: ["en", "hi", "hinglish"])
    
    # Audio parameters
    sample_rate: int = 16000
    chunk_duration: float = 30.0  # Max chunk duration in seconds
    
    # Text-to-Speech
    enable_tts: bool = True
    tts_engine: str = "gtts"  # gtts, pyttsx3, azure
    tts_voice: str = "en-IN"


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation"""
    # Vector database
    vector_db: str = "chromadb"  # chromadb, pinecone, weaviate, faiss
    collection_name: str = "medad_medical_kb"
    
    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval parameters
    retrieval_top_k: int = 5
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Knowledge sources
    knowledge_sources: List[str] = field(default_factory=lambda: [
        "medicine_database",
        "drug_interactions",
        "clinical_guidelines",
        "symptom_ontology"
    ])
    
    # LLM for generation
    generation_model: str = "gemini-2.0-flash"
    max_context_tokens: int = 4096
    temperature: float = 0.3


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualization"""
    # 3D Anatomy
    enable_3d_anatomy: bool = True
    anatomy_model_format: str = "gltf"  # gltf, obj, stl
    default_anatomy_model: str = "human_body"
    
    # Drug interaction graphs
    enable_interaction_graph: bool = True
    graph_layout: str = "force-directed"  # force-directed, hierarchical, circular
    max_nodes: int = 100
    
    # Plotly settings
    plotly_template: str = "plotly_white"
    animation_duration: int = 500


@dataclass
class MedAdConfig:
    """Master configuration for MedAd 2.0"""
    # API Keys (from environment)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    huggingface_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    
    # Global settings
    app_name: str = "MedAd 2.0"
    app_version: str = "2.0.0"
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Model size (affects all sub-configs)
    model_size: ModelSize = ModelSize.MEDIUM
    
    # Search mode
    search_mode: SearchMode = SearchMode.HYBRID
    
    # Sub-configurations
    semantic: SemanticEngineConfig = field(default_factory=SemanticEngineConfig)
    hinglish: HinglishConfig = field(default_factory=HinglishConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Data paths
    medicine_db_path: str = "all_medicine databased.csv"
    supplementary_db_path: str = "medicine_dataset.csv"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/medad.log"
    
    @classmethod
    def from_env(cls) -> "MedAdConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if os.getenv("MODEL_SIZE"):
            config.model_size = ModelSize(os.getenv("MODEL_SIZE"))
            
        if os.getenv("SEARCH_MODE"):
            config.search_mode = SearchMode(os.getenv("SEARCH_MODE"))
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)
    
    def apply_model_size_defaults(self):
        """Apply model size-based defaults to all sub-configs"""
        if self.model_size == ModelSize.NANO:
            self.semantic.primary_model = "all-MiniLM-L6-v2"
            self.semantic.embedding_dim = 384
            self.voice.whisper_model = "tiny"
            
        elif self.model_size == ModelSize.SMALL:
            self.semantic.primary_model = "all-MiniLM-L6-v2"
            self.voice.whisper_model = "base"
            
        elif self.model_size == ModelSize.LARGE:
            self.semantic.primary_model = "dmis-lab/biobert-large-cased-v1.1"
            self.semantic.embedding_dim = 1024
            self.voice.whisper_model = "medium"


# Global configuration instance
_config: Optional[MedAdConfig] = None


def get_config() -> MedAdConfig:
    """Get or create the global configuration instance"""
    global _config
    if _config is None:
        _config = MedAdConfig.from_env()
    return _config


def set_config(config: MedAdConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config
