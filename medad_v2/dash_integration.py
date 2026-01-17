"""
MEDAD 2.0 - Dash Integration Layer
=====================================

Bridge module connecting MedAd 2.0 backend with Dash frontend.

Provides:
- Callback wrappers for async functions
- UI component generators
- State management
- API endpoint handlers
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import base64
import json
import logging

from dash import html, dcc
import plotly.graph_objects as go

from .core.orchestrator import MedAdOrchestrator, MultimodalInput
from .core.config import MedAdConfig, get_config

logger = logging.getLogger("DashIntegration")

# Global orchestrator instance
_orchestrator: Optional[MedAdOrchestrator] = None


def get_orchestrator() -> MedAdOrchestrator:
    """Get or create the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MedAdOrchestrator()
    return _orchestrator


def async_to_sync(async_func: Callable) -> Callable:
    """Decorator to convert async functions for Dash callbacks"""
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


class MedAdDashIntegration:
    """
    Integration layer for MedAd 2.0 with Dash.
    
    Usage in web.py:
        from medad_v2.dash_integration import MedAdDashIntegration
        
        integration = MedAdDashIntegration()
        integration.initialize()
        
        @app.callback(...)
        def search_callback(query):
            return integration.search_medicines(query)
    """
    
    def __init__(self, config: Optional[MedAdConfig] = None):
        """Initialize the integration layer"""
        self.config = config or get_config()
        self.orchestrator = get_orchestrator()
        self._initialized = False
        
        # Legacy mode flag - use old TF-IDF if transformer loading fails
        self._legacy_mode = False
        
        logger.info("MedAdDashIntegration initialized")
    
    def initialize(self, components: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Initialize MedAd 2.0 components.
        
        Args:
            components: List of components to initialize
        
        Returns:
            Initialization status for each component
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self.orchestrator.initialize(components)
            )
            self._initialized = any(results.values())
            
            # Enable legacy mode if semantic engine failed
            if not results.get('semantic', False):
                self._legacy_mode = True
                logger.warning("Semantic engine failed, using legacy TF-IDF mode")
            
            return results
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._legacy_mode = True
            return {}
        finally:
            loop.close()
    
    def search_medicines(
        self,
        query: str,
        top_k: int = 15,
        language_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for medicines using MedAd 2.0 pipeline.
        
        Args:
            query: Symptom description
            top_k: Number of results
            language_hint: Optional language hint
        
        Returns:
            Dict with medicines, health_advice, and metadata
        """
        if self._legacy_mode or not self._initialized:
            # Return empty result - let legacy code handle it
            return {
                "medicines": [],
                "health_advice": None,
                "processed_query": query,
                "use_legacy": True
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.orchestrator.process_query(
                    query=query,
                    language_hint=language_hint
                )
            )
            
            return {
                "medicines": result.medicines[:top_k],
                "health_advice": result.health_advice,
                "processed_query": result.processed_query,
                "language_detected": result.language_detected,
                "confidence": result.confidence_score,
                "processing_time_ms": result.processing_time_ms,
                "use_legacy": False
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "medicines": [],
                "health_advice": None,
                "processed_query": query,
                "use_legacy": True,
                "error": str(e)
            }
        finally:
            loop.close()
    
    def process_multimodal(
        self,
        text: Optional[str] = None,
        image_data: Optional[str] = None,  # Base64 encoded
        audio_data: Optional[str] = None   # Base64 encoded
    ) -> Dict[str, Any]:
        """
        Process multimodal input (text + image + audio).
        
        Args:
            text: Text query
            image_data: Base64 encoded image
            audio_data: Base64 encoded audio
        
        Returns:
            Combined analysis results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Decode inputs
            image_bytes = base64.b64decode(image_data) if image_data else None
            audio_bytes = base64.b64decode(audio_data) if audio_data else None
            
            input_data = MultimodalInput(
                text=text,
                image=image_bytes,
                audio=audio_bytes
            )
            
            result = loop.run_until_complete(
                self.orchestrator.process_multimodal(input_data)
            )
            
            return {
                "medicines": result.medicines,
                "health_advice": result.health_advice,
                "image_analysis": result.image_analysis,
                "processed_query": result.processed_query,
                "language_detected": result.language_detected,
                "confidence": result.confidence_score,
                "warnings": result.warnings
            }
        except Exception as e:
            logger.error(f"Multimodal processing failed: {e}")
            return {"error": str(e)}
        finally:
            loop.close()
    
    def analyze_skin_image(self, image_data: str) -> Dict[str, Any]:
        """
        Analyze a skin condition image.
        
        Args:
            image_data: Base64 encoded image (with or without data URI prefix)
        
        Returns:
            Analysis results with detected conditions
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Handle data URI format
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            result = loop.run_until_complete(
                self.orchestrator.analyze_image(image_bytes)
            )
            
            # Handle both dict and object results
            if isinstance(result, dict):
                # Check for error
                if result.get('error'):
                    return {"error": result['error']}
                
                conditions = result.get('conditions', [])
                primary = result.get('primary_condition')
                image_quality = result.get('image_quality', 'unknown')
                warnings = result.get('warnings', [])
                disclaimer = result.get('disclaimer', 'Consult a healthcare professional.')
                
                # If no conditions but has warnings, it might be a non-medical image
                if not conditions and warnings:
                    return {
                        "conditions": [],
                        "primary_condition": None,
                        "image_quality": image_quality,
                        "warnings": warnings,
                        "disclaimer": disclaimer,
                        "message": warnings[0] if warnings else "No conditions detected"
                    }
                
                return {
                    "conditions": [
                        {
                            "name": c.get('name') if isinstance(c, dict) else c.name,
                            "confidence": c.get('confidence', 0) if isinstance(c, dict) else c.confidence,
                            "severity": c.get('severity', 'unknown') if isinstance(c, dict) else c.severity,
                            "description": c.get('description', '') if isinstance(c, dict) else c.description,
                            "treatments": c.get('common_treatments', []) if isinstance(c, dict) else getattr(c, 'common_treatments', []),
                            "seek_attention": c.get('seek_medical_attention', False) if isinstance(c, dict) else getattr(c, 'seek_medical_attention', False)
                        }
                        for c in conditions
                    ],
                    "primary_condition": primary.get('name') if isinstance(primary, dict) else (primary.name if primary else None),
                    "image_quality": image_quality,
                    "warnings": warnings,
                    "disclaimer": disclaimer
                }
            else:
                # Handle object-style result (ImageAnalysisResult)
                warnings = getattr(result, 'warnings', [])
                conditions = getattr(result, 'conditions', [])
                
                # If no conditions but has warnings, it might be a non-medical image
                if not conditions and warnings:
                    return {
                        "conditions": [],
                        "primary_condition": None,
                        "image_quality": getattr(result, 'image_quality', 'unknown'),
                        "warnings": warnings,
                        "disclaimer": getattr(result, 'disclaimer', 'Consult a healthcare professional.'),
                        "message": warnings[0] if warnings else "No conditions detected"
                    }
                
                return {
                    "conditions": [
                        {
                            "name": c.name,
                            "confidence": c.confidence,
                            "severity": c.severity,
                            "description": c.description,
                            "treatments": getattr(c, 'common_treatments', []),
                            "seek_attention": getattr(c, 'seek_medical_attention', False)
                        }
                        for c in conditions
                    ],
                    "primary_condition": result.primary_condition.name if hasattr(result, 'primary_condition') and result.primary_condition else None,
                    "image_quality": getattr(result, 'image_quality', 'unknown'),
                    "warnings": warnings,
                    "disclaimer": getattr(result, 'disclaimer', 'Consult a healthcare professional.')
                }
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"error": str(e)}
        finally:
            loop.close()
    
    def get_3d_visualization(
        self,
        symptom: str,
        medicines: List[Dict[str, Any]],
        viz_type: str = "drug_interaction"
    ) -> go.Figure:
        """
        Generate 3D visualization.
        
        Args:
            symptom: The symptom/condition
            medicines: List of medicines
            viz_type: Type of visualization
        
        Returns:
            Plotly Figure object
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.orchestrator.get_3d_visualization(
                    symptom=symptom,
                    medicines=medicines,
                    viz_type=viz_type
                )
            )
            
            return result.plotly_figure if result.plotly_figure else go.Figure()
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return go.Figure()
        finally:
            loop.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return self.orchestrator.get_statistics()


def create_image_upload_component() -> html.Div:
    """Create the image upload UI component for skin analysis"""
    return html.Div([
        html.H4("🖼️ Skin Condition Analyzer", style={"color": "#00695C"}),
        html.P("Upload an image of your skin condition for AI analysis",
               style={"color": "#666", "fontSize": "14px"}),
        dcc.Upload(
            id='skin-image-upload',
            children=html.Div([
                html.I(className="fas fa-camera", style={"fontSize": "24px", "marginRight": "10px"}),
                'Drag and Drop or Click to Upload Image'
            ]),
            style={
                'width': '100%',
                'height': '100px',
                'lineHeight': '100px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '10px',
                'borderColor': '#00695C',
                'textAlign': 'center',
                'cursor': 'pointer',
                'backgroundColor': 'rgba(178, 223, 219, 0.3)'
            },
            accept='image/*'
        ),
        html.Div(id='skin-analysis-output'),
        html.P("⚠️ For educational purposes only. Consult a dermatologist for diagnosis.",
               style={"color": "#D32F2F", "fontSize": "12px", "marginTop": "10px"})
    ], style={
        "padding": "20px",
        "backgroundColor": "white",
        "borderRadius": "15px",
        "marginTop": "20px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
    })


def create_voice_input_component() -> html.Div:
    """Create the voice input UI component"""
    return html.Div([
        html.H4("🎤 Voice Search", style={"color": "#00695C"}),
        html.P("Speak your symptoms in English, Hindi, or Hinglish",
               style={"color": "#666", "fontSize": "14px"}),
        html.Button(
            [html.I(className="fas fa-microphone", style={"marginRight": "10px"}),
             "Start Voice Input"],
            id='voice-input-button',
            style={
                'width': '100%',
                'padding': '15px',
                'backgroundColor': '#00695C',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'cursor': 'pointer',
                'fontSize': '16px'
            }
        ),
        html.Div(id='voice-transcription-output',
                 style={"marginTop": "15px", "padding": "10px"})
    ], style={
        "padding": "20px",
        "backgroundColor": "white",
        "borderRadius": "15px",
        "marginTop": "20px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
    })


def create_3d_visualization_component(graph_id: str = "drug-3d-graph") -> html.Div:
    """Create the 3D visualization container"""
    return html.Div([
        html.H4("🧬 3D Drug Interaction Network", style={"color": "#00695C"}),
        dcc.Graph(
            id=graph_id,
            style={"height": "500px"},
            config={
                'displayModeBar': True,
                'scrollZoom': True
            }
        )
    ], style={
        "padding": "20px",
        "backgroundColor": "white",
        "borderRadius": "15px",
        "marginTop": "20px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
    })


# JavaScript for browser-based voice recognition (to be injected in app)
VOICE_RECOGNITION_JS = """
<script>
let recognition;
let isRecording = false;

function initVoiceRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-IN'; // Indian English with Hindi support
        
        recognition.onresult = function(event) {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                transcript += event.results[i][0].transcript;
            }
            document.getElementById('voice-transcription').value = transcript;
            // Trigger Dash callback
            document.getElementById('symptom-input').value = transcript;
            document.getElementById('symptom-input').dispatchEvent(new Event('input'));
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
        };
    }
}

function toggleRecording() {
    if (!recognition) {
        initVoiceRecognition();
    }
    
    if (isRecording) {
        recognition.stop();
        isRecording = false;
        document.getElementById('voice-input-button').innerHTML = 
            '<i class="fas fa-microphone"></i> Start Voice Input';
    } else {
        recognition.start();
        isRecording = true;
        document.getElementById('voice-input-button').innerHTML = 
            '<i class="fas fa-stop-circle"></i> Stop Recording';
    }
}

document.addEventListener('DOMContentLoaded', initVoiceRecognition);
</script>
"""
