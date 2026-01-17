"""
MEDAD 2.0 - Main Orchestrator
================================

Central orchestration module that coordinates all MedAd 2.0 subsystems:
- Semantic search engine (BioBERT/ClinicalBERT)
- Hinglish NLP processor
- Computer vision for dermatology
- Voice interface
- RAG knowledge retrieval
- 3D visualization

This module provides a unified API for the Dash web interface.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from .config import MedAdConfig, get_config, SearchMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedAdOrchestrator")


@dataclass
class DiagnosticResult:
    """Structured result from diagnostic pipeline"""
    query: str
    processed_query: str
    language_detected: str
    medicines: List[Dict[str, Any]]
    health_advice: Optional[str]
    image_analysis: Optional[Dict[str, Any]]
    confidence_score: float
    search_mode: str
    processing_time_ms: float
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class MultimodalInput:
    """Container for multimodal user input"""
    text: Optional[str] = None
    image: Optional[bytes] = None
    audio: Optional[bytes] = None
    language_hint: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class MedAdOrchestrator:
    """
    Main orchestrator for MedAd 2.0 multimodal medical assistant.
    
    Coordinates:
    1. Language detection & Hinglish processing
    2. Transformer-based semantic symptom understanding
    3. Computer vision for skin condition analysis
    4. Voice-to-text conversion
    5. RAG-enhanced knowledge retrieval
    6. Medicine recommendation ranking
    7. 3D visualization generation
    
    Usage:
        orchestrator = MedAdOrchestrator()
        await orchestrator.initialize()
        result = await orchestrator.process_query("sar me bahut dard hai")
    """
    
    def __init__(self, config: Optional[MedAdConfig] = None):
        """Initialize the orchestrator with configuration"""
        self.config = config or get_config()
        self.initialized = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Lazy-loaded components
        self._semantic_engine = None
        self._hinglish_processor = None
        self._vision_analyzer = None
        self._voice_interface = None
        self._rag_retriever = None
        self._visualizer = None
        
        # Statistics
        self._query_count = 0
        self._total_processing_time = 0.0
        
        logger.info(f"MedAdOrchestrator initialized with config: {self.config.app_name}")
    
    async def initialize(self, components: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Initialize all or specified components.
        
        Args:
            components: List of components to initialize. 
                       None = all components.
                       Options: ['semantic', 'hinglish', 'vision', 'voice', 'rag', 'viz']
        
        Returns:
            Dictionary mapping component names to initialization success status
        """
        results = {}
        components = components or ['semantic', 'hinglish', 'vision', 'voice', 'rag', 'viz']
        
        logger.info(f"Initializing components: {components}")
        
        if 'semantic' in components:
            results['semantic'] = await self._init_semantic_engine()
            
        if 'hinglish' in components:
            results['hinglish'] = await self._init_hinglish_processor()
            
        if 'vision' in components:
            results['vision'] = await self._init_vision_analyzer()
            
        if 'voice' in components:
            results['voice'] = await self._init_voice_interface()
            
        if 'rag' in components:
            results['rag'] = await self._init_rag_retriever()
            
        if 'viz' in components:
            results['viz'] = await self._init_visualizer()
        
        self.initialized = all(results.values())
        logger.info(f"Initialization complete. Status: {results}")
        
        return results
    
    async def _init_semantic_engine(self) -> bool:
        """Initialize the transformer-based semantic search engine"""
        try:
            from ..semantic_engine.transformer_search import TransformerSearchEngine
            self._semantic_engine = TransformerSearchEngine(self.config.semantic)
            await self._semantic_engine.load_model()
            logger.info("✅ Semantic engine initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Semantic engine failed to initialize: {e}")
            return False
    
    async def _init_hinglish_processor(self) -> bool:
        """Initialize the Hinglish NLP processor"""
        try:
            from ..nlp.hinglish_processor import HinglishProcessor
            self._hinglish_processor = HinglishProcessor(self.config.hinglish)
            await self._hinglish_processor.load_mappings()
            logger.info("✅ Hinglish processor initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Hinglish processor failed to initialize: {e}")
            return False
    
    async def _init_vision_analyzer(self) -> bool:
        """Initialize the computer vision module"""
        try:
            from ..vision.derma_analyzer import DermaAnalyzer
            self._vision_analyzer = DermaAnalyzer(self.config.vision)
            await self._vision_analyzer.load_model()
            logger.info("✅ Vision analyzer initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Vision analyzer failed to initialize: {e}")
            return False
    
    async def _init_voice_interface(self) -> bool:
        """Initialize the voice interface"""
        try:
            from ..voice.speech_processor import SpeechProcessor
            self._voice_interface = SpeechProcessor(self.config.voice)
            logger.info("✅ Voice interface initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Voice interface failed to initialize: {e}")
            return False
    
    async def _init_rag_retriever(self) -> bool:
        """Initialize the RAG knowledge retriever"""
        try:
            from ..rag.knowledge_retriever import KnowledgeRetriever
            self._rag_retriever = KnowledgeRetriever(self.config.rag)
            await self._rag_retriever.initialize()
            logger.info("✅ RAG retriever initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ RAG retriever failed to initialize: {e}")
            return False
    
    async def _init_visualizer(self) -> bool:
        """Initialize the 3D visualization module"""
        try:
            from ..visualization.medical_viz import MedicalVisualizer
            self._visualizer = MedicalVisualizer(self.config.visualization)
            logger.info("✅ Visualizer initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Visualizer failed to initialize: {e}")
            return False
    
    async def process_multimodal(self, input_data: MultimodalInput) -> DiagnosticResult:
        """
        Process multimodal input (text + image + audio).
        
        This is the main entry point for the MedAd 2.0 pipeline.
        
        Args:
            input_data: MultimodalInput containing text, image, and/or audio
        
        Returns:
            DiagnosticResult with medicines, advice, and analysis
        """
        start_time = time.time()
        warnings = []
        
        # Stage 1: Convert audio to text if present
        text_input = input_data.text or ""
        if input_data.audio and self._voice_interface:
            try:
                transcription = await self._voice_interface.transcribe(input_data.audio)
                text_input = f"{text_input} {transcription.text}".strip()
                logger.info(f"🎤 Audio transcribed: {transcription.text}")
            except Exception as e:
                warnings.append(f"Audio transcription failed: {e}")
        
        # Stage 2: Analyze image if present
        image_analysis = None
        if input_data.image and self._vision_analyzer:
            try:
                image_analysis = await self._vision_analyzer.analyze(input_data.image)
                # Append detected conditions to text query
                if image_analysis.get('conditions'):
                    conditions_text = " ".join([c['name'] for c in image_analysis['conditions'][:3]])
                    text_input = f"{text_input} {conditions_text}".strip()
                logger.info(f"🖼️ Image analyzed: {image_analysis.get('conditions', [])}")
            except Exception as e:
                warnings.append(f"Image analysis failed: {e}")
        
        # Stage 3: Process text through main pipeline
        result = await self.process_query(
            query=text_input,
            language_hint=input_data.language_hint,
            context=input_data.context
        )
        
        # Attach image analysis to result
        result.image_analysis = image_analysis
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.warnings.extend(warnings)
        
        return result
    
    async def process_query(
        self,
        query: str,
        language_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DiagnosticResult:
        """
        Process a text query through the diagnostic pipeline.
        
        Pipeline stages:
        1. Language detection & preprocessing
        2. Hinglish transliteration (if needed)
        3. Symptom entity extraction
        4. Semantic search for medicines
        5. RAG knowledge retrieval
        6. Health advice generation
        7. Result ranking and formatting
        
        Args:
            query: User's symptom description (can be English, Hindi, or Hinglish)
            language_hint: Optional language hint (en, hi, hinglish)
            context: Optional context from conversation history
        
        Returns:
            DiagnosticResult with ranked medicines and health advice
        """
        start_time = time.time()
        warnings = []
        metadata = {}
        
        # Initialize default values
        processed_query = query
        language_detected = language_hint or "en"
        
        # Stage 1: Language detection and Hinglish processing
        if self._hinglish_processor:
            try:
                lang_result = await self._hinglish_processor.process(query)
                processed_query = lang_result.normalized_text
                language_detected = lang_result.detected_language
                metadata['hinglish_entities'] = lang_result.extracted_entities
                metadata['phonetic_matches'] = lang_result.phonetic_matches
                logger.info(f"🌐 Language: {language_detected}, Processed: {processed_query}")
            except Exception as e:
                warnings.append(f"Hinglish processing failed: {e}")
                processed_query = query
        
        # Stage 2: Semantic search for medicines
        medicines = []
        confidence_score = 0.0
        search_mode = self.config.search_mode.value
        
        if self._semantic_engine:
            try:
                search_result = await self._semantic_engine.search(
                    query=processed_query,
                    top_k=self.config.semantic.top_k_results
                )
                medicines = search_result.results
                confidence_score = search_result.average_confidence
                metadata['semantic_scores'] = search_result.scores
                logger.info(f"💊 Found {len(medicines)} medicines with confidence {confidence_score:.2f}")
            except Exception as e:
                warnings.append(f"Semantic search failed: {e}")
        
        # Stage 3: RAG knowledge retrieval (if enabled)
        if self._rag_retriever and self.config.search_mode == SearchMode.RAG:
            try:
                rag_result = await self._rag_retriever.retrieve(
                    query=processed_query,
                    medicines=medicines[:5]
                )
                metadata['rag_context'] = rag_result.context
                metadata['rag_sources'] = rag_result.sources
            except Exception as e:
                warnings.append(f"RAG retrieval failed: {e}")
        
        # Stage 4: Generate health advice
        health_advice = None
        if self.config.gemini_api_key and medicines:
            try:
                health_advice = await self._generate_health_advice(
                    symptom=processed_query,
                    medicines=medicines[:5],
                    rag_context=metadata.get('rag_context')
                )
            except Exception as e:
                warnings.append(f"Health advice generation failed: {e}")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self._query_count += 1
        self._total_processing_time += processing_time
        
        return DiagnosticResult(
            query=query,
            processed_query=processed_query,
            language_detected=language_detected,
            medicines=medicines,
            health_advice=health_advice,
            image_analysis=None,
            confidence_score=confidence_score,
            search_mode=search_mode,
            processing_time_ms=processing_time,
            warnings=warnings,
            metadata=metadata
        )
    
    async def _generate_health_advice(
        self,
        symptom: str,
        medicines: List[Dict[str, Any]],
        rag_context: Optional[str] = None
    ) -> str:
        """Generate health advice using Gemini AI with optional RAG context"""
        import urllib.request
        import json
        
        medicine_list = ", ".join([m.get('Medicine Name', '') for m in medicines[:5]])
        
        # Build prompt with RAG context if available
        context_section = ""
        if rag_context:
            context_section = f"\n\nRelevant medical knowledge:\n{rag_context}\n"
        
        prompt = f"""You are a helpful medical AI assistant. The user has symptoms: "{symptom}".
        Based on our database, we found these medicines: {medicine_list}.
        {context_section}
        Please provide:
        1. Brief health tip (1-2 sentences)
        2. When to see a doctor
        3. Home remedies (if applicable)
        
        Keep response under 100 words. Be professional and caring."""
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.config.gemini_api_key}"
            
            data = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 200
                }
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=data, headers={
                'Content-Type': 'application/json'
            })
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: urllib.request.urlopen(req, timeout=10)
            )
            
            result = json.loads(response.read().decode('utf-8'))
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            
        return None
    
    async def analyze_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze a skin condition image.
        
        Args:
            image_data: Image bytes (JPEG/PNG)
        
        Returns:
            Analysis result with detected conditions and confidence scores
        """
        if not self._vision_analyzer:
            return {"error": "Vision analyzer not initialized"}
        
        # Pass Gemini API key for enhanced analysis
        return await self._vision_analyzer.analyze(
            image_data, 
            api_key=self.config.gemini_api_key
        )
    
    async def get_3d_visualization(
        self,
        symptom: str,
        medicines: List[Dict[str, Any]],
        viz_type: str = "drug_interaction"
    ) -> Dict[str, Any]:
        """
        Generate 3D visualization data.
        
        Args:
            symptom: The symptom/condition
            medicines: List of related medicines
            viz_type: Type of visualization (drug_interaction, anatomy, prevalence)
        
        Returns:
            Visualization data for frontend rendering
        """
        if not self._visualizer:
            return {"error": "Visualizer not initialized"}
        
        return await self._visualizer.generate(
            symptom=symptom,
            medicines=medicines,
            viz_type=viz_type
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "total_queries": self._query_count,
            "total_processing_time_ms": self._total_processing_time,
            "average_processing_time_ms": (
                self._total_processing_time / self._query_count 
                if self._query_count > 0 else 0
            ),
            "initialized_components": {
                "semantic": self._semantic_engine is not None,
                "hinglish": self._hinglish_processor is not None,
                "vision": self._vision_analyzer is not None,
                "voice": self._voice_interface is not None,
                "rag": self._rag_retriever is not None,
                "visualizer": self._visualizer is not None
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down MedAdOrchestrator...")
        self._executor.shutdown(wait=True)
        
        # Cleanup components
        if self._semantic_engine:
            await self._semantic_engine.cleanup()
        if self._rag_retriever:
            await self._rag_retriever.cleanup()
            
        self.initialized = False
        logger.info("Shutdown complete")
