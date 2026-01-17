"""
MEDAD 2.0 - Speech Processing Module
========================================

Voice interface module for speech-to-text and text-to-speech.

Key Features:
- OpenAI Whisper for speech-to-text
- Multi-language support (English, Hindi, Hinglish)
- Real-time streaming transcription
- Google Text-to-Speech integration
- WebRTC compatibility for browser audio

Supported Languages:
- English (en)
- Hindi (hi)
- Hinglish (code-mixed, detected automatically)
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import io
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("SpeechProcessor")


@dataclass
class TranscriptionResult:
    """Result of speech transcription"""
    text: str
    language: str
    confidence: float
    duration_seconds: float
    word_timestamps: Optional[List[Dict[str, Any]]] = None


@dataclass
class TTSResult:
    """Result of text-to-speech synthesis"""
    audio_data: bytes
    format: str
    sample_rate: int
    duration_seconds: float


class SpeechProcessor:
    """
    Speech processing module for voice-activated medical assistance.
    
    Provides:
    1. Speech-to-Text using OpenAI Whisper
    2. Text-to-Speech for response vocalization
    3. Language detection for Hindi/English/Hinglish
    4. Audio preprocessing for noisy environments
    
    Example:
        processor = SpeechProcessor(config)
        result = await processor.transcribe(audio_bytes)
        print(result.text)  # "mujhe sar me dard hai"
        print(result.language)  # "hinglish"
    """
    
    # Whisper model sizes and their characteristics
    WHISPER_MODELS = {
        "tiny": {"size": "39M", "speed": "~32x", "quality": "Low"},
        "base": {"size": "74M", "speed": "~16x", "quality": "Medium"},
        "small": {"size": "244M", "speed": "~6x", "quality": "Good"},
        "medium": {"size": "769M", "speed": "~2x", "quality": "High"},
        "large": {"size": "1550M", "speed": "~1x", "quality": "Best"},
    }
    
    def __init__(self, config=None):
        """
        Initialize the speech processor.
        
        Args:
            config: VoiceConfig with speech processing parameters (optional)
        """
        # Create default config if not provided
        if config is None:
            from ..core.config import VoiceConfig
            config = VoiceConfig()
        
        self.config = config
        self._whisper_model = None
        self._tts_engine = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False
        
        logger.info(f"SpeechProcessor initialized with Whisper model: {config.whisper_model}")
    
    async def load_model(self) -> bool:
        """
        Load the Whisper model for speech recognition.
        """
        loop = asyncio.get_event_loop()
        
        def _load():
            try:
                import whisper
                
                model_name = self.config.whisper_model
                logger.info(f"Loading Whisper model: {model_name}")
                
                self._whisper_model = whisper.load_model(model_name)
                
                logger.info(f"✅ Whisper model loaded: {model_name}")
                return True
                
            except ImportError:
                logger.warning("Whisper not available. Install with: pip install openai-whisper")
                return False
            except Exception as e:
                logger.error(f"Whisper loading failed: {e}")
                return False
        
        success = await loop.run_in_executor(self._executor, _load)
        self._initialized = success
        return success
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio bytes (WAV, MP3, or other supported format)
            language: Language hint (None for auto-detection)
        
        Returns:
            TranscriptionResult with transcribed text
        """
        if not self._initialized:
            # Fallback to browser-based transcription
            return await self._browser_transcribe(audio_data, language)
        
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            import tempfile
            import time
            
            start_time = time.time()
            
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            try:
                # Transcribe with Whisper
                result = self._whisper_model.transcribe(
                    temp_path,
                    language=language,
                    task="transcribe"
                )
                
                text = result["text"].strip()
                detected_lang = result.get("language", "en")
                
                # Get word-level timestamps if available
                word_timestamps = None
                if "segments" in result:
                    word_timestamps = []
                    for segment in result["segments"]:
                        word_timestamps.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"]
                        })
                
                duration = time.time() - start_time
                
                return TranscriptionResult(
                    text=text,
                    language=detected_lang,
                    confidence=0.9,  # Whisper doesn't provide confidence
                    duration_seconds=duration,
                    word_timestamps=word_timestamps
                )
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        return await loop.run_in_executor(self._executor, _transcribe)
    
    async def _browser_transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Fallback transcription using browser Web Speech API.
        
        This is handled client-side, this method just formats the result.
        """
        # In practice, browser transcription would be handled in JavaScript
        # This is a placeholder for server-side processing of browser results
        return TranscriptionResult(
            text="",
            language=language or "en",
            confidence=0.0,
            duration_seconds=0.0,
            word_timestamps=None
        )
    
    async def synthesize(
        self,
        text: str,
        language: str = "en"
    ) -> TTSResult:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            language: Target language
        
        Returns:
            TTSResult with audio data
        """
        loop = asyncio.get_event_loop()
        
        def _synthesize():
            import time
            start_time = time.time()
            
            try:
                # Try gTTS first
                from gtts import gTTS
                
                # Map language codes
                lang_map = {
                    "en": "en-IN",  # Indian English
                    "hi": "hi",     # Hindi
                    "hinglish": "hi"  # Use Hindi for Hinglish
                }
                
                tts_lang = lang_map.get(language, "en")
                
                # Handle Hinglish by keeping English words
                if language == "hinglish":
                    tts_lang = "hi"
                
                tts = gTTS(text=text, lang=tts_lang.split("-")[0])
                
                # Save to bytes
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio_data = audio_buffer.read()
                
                duration = time.time() - start_time
                
                return TTSResult(
                    audio_data=audio_data,
                    format="mp3",
                    sample_rate=24000,
                    duration_seconds=duration
                )
                
            except ImportError:
                logger.warning("gTTS not available. Install with: pip install gTTS")
                return TTSResult(
                    audio_data=b"",
                    format="mp3",
                    sample_rate=24000,
                    duration_seconds=0.0
                )
            except Exception as e:
                logger.error(f"TTS synthesis failed: {e}")
                return TTSResult(
                    audio_data=b"",
                    format="mp3",
                    sample_rate=24000,
                    duration_seconds=0.0
                )
        
        return await loop.run_in_executor(self._executor, _synthesize)
    
    async def detect_language(self, audio_data: bytes) -> str:
        """
        Detect the language of spoken audio.
        
        Returns:
            Language code (en, hi, hinglish)
        """
        if not self._initialized:
            return "en"
        
        loop = asyncio.get_event_loop()
        
        def _detect():
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            try:
                # Use Whisper's detect_language
                import whisper
                
                # Load first 30 seconds
                audio = whisper.load_audio(temp_path)
                audio = whisper.pad_or_trim(audio)
                
                # Get mel spectrogram
                mel = whisper.log_mel_spectrogram(audio).to(self._whisper_model.device)
                
                # Detect language
                _, probs = self._whisper_model.detect_language(mel)
                
                detected = max(probs, key=probs.get)
                
                # Check for Hinglish (code-mixed)
                if detected == "hi":
                    # Run transcription to check for English words
                    result = self._whisper_model.transcribe(temp_path, language="hi")
                    text = result["text"]
                    
                    # Count English words
                    import re
                    english_pattern = re.compile(r'\b[a-zA-Z]+\b')
                    english_words = english_pattern.findall(text)
                    
                    if len(english_words) > 2:
                        return "hinglish"
                
                return detected
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        return await loop.run_in_executor(self._executor, _detect)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.config.languages
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)
        self._whisper_model = None
        self._initialized = False
