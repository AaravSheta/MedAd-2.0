"""
MEDAD 2.0 - Hinglish NLP Processor
=====================================

Advanced natural language processor for Hinglish (Hindi-English code-mixed)
and Romanized Hindi medical symptom descriptions.

Key Features:
- Phonetic matching (sar dard → headache)
- Code-mixed language detection
- Romanized Hindi transliteration
- Medical term normalization
- Fuzzy symptom matching

Example conversions:
- "sar me bahut dard hai" → "severe headache"
- "pet me infection" → "stomach infection"
- "bukhar aur khansi" → "fever and cough"
- "gala dard" → "sore throat"
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("HinglishProcessor")


@dataclass
class ProcessingResult:
    """Result of Hinglish text processing"""
    original_text: str
    normalized_text: str
    detected_language: str  # 'en', 'hi', 'hinglish'
    extracted_entities: List[Dict[str, str]]
    phonetic_matches: List[Tuple[str, str, float]]  # (original, matched, confidence)
    confidence: float


# Comprehensive Hinglish medical term mappings
# Format: Romanized Hindi → English medical term
HINGLISH_MEDICAL_TERMS = {
    # Head-related
    "sar": "head",
    "sir": "head",
    "sar dard": "headache",
    "sir dard": "headache",
    "sar me dard": "headache",
    "sir me dard": "headache",
    "sar bahut dard": "severe headache",
    "maatha": "forehead",
    "chakkar": "dizziness",
    "chakkar aa raha": "feeling dizzy",
    "chakkar aana": "dizziness",
    
    # Fever-related
    "bukhar": "fever",
    "bukhaar": "fever",
    "tez bukhar": "high fever",
    "halka bukhar": "mild fever",
    "badan garam": "body hot fever",
    "thandi": "chills",
    "kaanpna": "shivering",
    
    # Respiratory
    "khansi": "cough",
    "khaansi": "cough",
    "sukhi khansi": "dry cough",
    "geeli khansi": "wet cough",
    "balgam": "phlegm",
    "saans": "breath",
    "saans lene me dikkat": "breathing difficulty",
    "saans phoolna": "shortness of breath",
    "naak": "nose",
    "naak band": "nasal congestion",
    "naak behna": "runny nose",
    "chhink": "sneeze",
    "nazla": "cold",
    "zukaam": "cold",
    "jukaam": "cold",
    
    # Throat-related
    "gala": "throat",
    "gale me dard": "sore throat",
    "gala dard": "sore throat",
    "gala kharab": "throat problem",
    "awaaz": "voice",
    "awaaz baithi": "hoarse voice",
    
    # Stomach/Digestive
    "pet": "stomach",
    "pet dard": "stomach pain",
    "pet me dard": "abdominal pain",
    "pet kharab": "upset stomach",
    "ulti": "vomiting",
    "ji michlana": "nausea",
    "matli": "nausea",
    "dast": "diarrhea",
    "loose motion": "diarrhea",
    "kabz": "constipation",
    "qabz": "constipation",
    "gas": "gas",
    "gais": "gas",
    "bloating": "bloating",
    "pet phoolna": "bloating",
    "acidity": "acidity",
    "seene me jalan": "heartburn",
    "khatta dakar": "acid reflux",
    
    # Pain-related
    "dard": "pain",
    "dard hona": "pain",
    "bahut dard": "severe pain",
    "halka dard": "mild pain",
    "teez dard": "sharp pain",
    "kamar dard": "back pain",
    "peeth dard": "back pain",
    "ghutne me dard": "knee pain",
    "ghutna dard": "knee pain",
    "jodo me dard": "joint pain",
    "jod dard": "joint pain",
    "haath dard": "hand pain",
    "pair dard": "leg pain",
    "taang dard": "leg pain",
    "dant dard": "toothache",
    "daant dard": "toothache",
    
    # Skin-related
    "chamdi": "skin",
    "khujli": "itching",
    "kharish": "itching",
    "daane": "rash",
    "dane": "pimples",
    "muhase": "acne",
    "pimple": "acne",
    "jalan": "burning",
    "sujan": "swelling",
    "soojan": "swelling",
    "lal hona": "redness",
    "allergy": "allergy",
    "reaction": "allergic reaction",
    
    # Eyes
    "aankh": "eye",
    "aankhon me dard": "eye pain",
    "aankh lal": "red eye",
    "aankh se paani": "watery eyes",
    "dhundla dikhna": "blurred vision",
    
    # Ears
    "kaan": "ear",
    "kaan dard": "ear pain",
    "kaan me dard": "ear pain",
    "kaan bajana": "tinnitus",
    "sunai kam dena": "hearing loss",
    
    # General symptoms
    "kamzori": "weakness",
    "thakan": "fatigue",
    "neend nahi aana": "insomnia",
    "neend na aana": "sleeplessness",
    "bhukh nahi": "loss of appetite",
    "pyaas": "thirst",
    "peshab": "urination",
    "peshab me jalan": "burning urination",
    "weight badna": "weight gain",
    "weight kam hona": "weight loss",
    "pasina": "sweating",
    
    # Diseases
    "sugar": "diabetes",
    "madhumeh": "diabetes",
    "bp": "blood pressure",
    "uchh raktchap": "high blood pressure",
    "dil": "heart",
    "dil ki bimari": "heart disease",
    "kidney": "kidney",
    "gurde": "kidney",
    "liver": "liver",
    "jigar": "liver",
    "pilia": "jaundice",
    "peeliya": "jaundice",
    "tb": "tuberculosis",
    "cancer": "cancer",
    "tumor": "tumor",
    "infection": "infection",
    "sankraman": "infection",
    
    # Intensity modifiers
    "bahut": "very severe",
    "thoda": "mild slight",
    "kuch": "some",
    "zyada": "more excessive",
    "kam": "less mild",
    "tez": "sharp intense",
    "dhire dhire": "gradually",
    "achanak": "sudden",
    "lagatar": "continuous",
    "ruk ruk kar": "intermittent",
    
    # Time-related
    "subah": "morning",
    "shaam": "evening",
    "raat": "night",
    "din": "day",
    "hafta": "week",
    "mahina": "month",
    "kal se": "since yesterday",
    "kab se": "since when",
    "kitne din": "how many days",
}

# Phonetic equivalents for fuzzy matching
PHONETIC_EQUIVALENTS = {
    # Vowel variations
    'aa': 'a',
    'ee': 'i',
    'oo': 'u',
    'ai': 'e',
    'au': 'o',
    
    # Consonant variations
    'ph': 'f',
    'bh': 'b',
    'dh': 'd',
    'th': 't',
    'gh': 'g',
    'kh': 'k',
    'ch': 'c',
    'sh': 's',
    'zh': 'j',
    
    # Common misspellings
    'q': 'k',
    'x': 'ks',
    'z': 's',
}


class HinglishProcessor:
    """
    Advanced Hinglish NLP processor for medical symptom understanding.
    
    Handles:
    1. Romanized Hindi detection and transliteration
    2. Code-mixed Hindi-English text processing
    3. Phonetic matching for spelling variations
    4. Medical entity extraction
    
    Example:
        processor = HinglishProcessor(config)
        result = await processor.process("sar me bahut dard hai aur bukhar bhi")
        # result.normalized_text = "severe headache and fever"
    """
    
    def __init__(self, config=None):
        """
        Initialize the Hinglish processor.
        
        Args:
            config: HinglishConfig with processing parameters (optional)
        """
        # Create default config if not provided
        if config is None:
            from ..core.config import HinglishConfig
            config = HinglishConfig()
        
        self.config = config
        self._term_mappings = HINGLISH_MEDICAL_TERMS.copy()
        self._phonetic_engine = None
        self._transliterator = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("HinglishProcessor initialized")
    
    async def load_mappings(self, custom_mappings_file: Optional[str] = None):
        """
        Load custom Hinglish mappings from file.
        
        Args:
            custom_mappings_file: Path to JSON file with custom mappings
        """
        if custom_mappings_file:
            try:
                import json
                with open(custom_mappings_file, 'r', encoding='utf-8') as f:
                    custom_mappings = json.load(f)
                    self._term_mappings.update(custom_mappings)
                    logger.info(f"Loaded {len(custom_mappings)} custom mappings")
            except Exception as e:
                logger.warning(f"Failed to load custom mappings: {e}")
        
        # Initialize phonetic engine
        from .phonetic_engine import PhoneticEngine
        self._phonetic_engine = PhoneticEngine()
        
        logger.info(f"Loaded {len(self._term_mappings)} Hinglish medical term mappings")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Returns:
            'en' - English
            'hi' - Hindi (Devanagari script)
            'hinglish' - Mixed Hindi-English or Romanized Hindi
        """
        # Check for Devanagari script
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        if devanagari_pattern.search(text):
            return 'hi'
        
        # Check for English-only
        english_pattern = re.compile(r'^[a-zA-Z\s.,!?0-9]+$')
        if english_pattern.match(text):
            # Check if any Hinglish terms present
            text_lower = text.lower()
            hinglish_count = sum(1 for term in self._term_mappings if term in text_lower)
            if hinglish_count > 0:
                return 'hinglish'
            return 'en'
        
        return 'hinglish'
    
    def normalize_phonetic(self, text: str) -> str:
        """
        Normalize phonetic variations in Romanized Hindi.
        
        Handles common spelling variations:
        - bukhar/bukhaar → fever
        - khansi/khaansi → cough
        - sar/sir → head
        """
        normalized = text.lower()
        
        # Apply phonetic equivalents
        for variant, standard in PHONETIC_EQUIVALENTS.items():
            normalized = normalized.replace(variant, standard)
        
        return normalized
    
    def extract_hinglish_terms(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract and translate Hinglish medical terms.
        
        Returns list of (original_term, english_translation, confidence)
        """
        text_lower = text.lower()
        matches = []
        
        # Sort by length (longer terms first) to match phrases before words
        sorted_terms = sorted(self._term_mappings.keys(), key=len, reverse=True)
        
        used_positions = set()
        
        for hinglish_term in sorted_terms:
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(hinglish_term, start)
                if pos == -1:
                    break
                
                end_pos = pos + len(hinglish_term)
                
                # Check if this position overlaps with already matched terms
                if not any(p in used_positions for p in range(pos, end_pos)):
                    english = self._term_mappings[hinglish_term]
                    confidence = 0.95 if len(hinglish_term) > 5 else 0.85
                    matches.append((hinglish_term, english, confidence))
                    
                    # Mark positions as used
                    for p in range(pos, end_pos):
                        used_positions.add(p)
                
                start = end_pos
        
        return matches
    
    def fuzzy_match_term(self, term: str, threshold: float = 0.7) -> Optional[Tuple[str, str, float]]:
        """
        Fuzzy match a term against known Hinglish terms.
        
        Uses phonetic normalization and edit distance.
        """
        from difflib import SequenceMatcher
        
        normalized_term = self.normalize_phonetic(term)
        best_match = None
        best_score = 0.0
        
        for hinglish_term, english in self._term_mappings.items():
            normalized_hinglish = self.normalize_phonetic(hinglish_term)
            
            # Calculate similarity
            score = SequenceMatcher(None, normalized_term, normalized_hinglish).ratio()
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = (hinglish_term, english, score)
        
        return best_match
    
    async def process(self, text: str) -> ProcessingResult:
        """
        Process Hinglish text and normalize to English medical terms.
        
        Pipeline:
        1. Language detection
        2. Phonetic normalization
        3. Term extraction and translation
        4. Fuzzy matching for unknown terms
        5. Result assembly
        
        Args:
            text: Input text (can be English, Hindi, or Hinglish)
        
        Returns:
            ProcessingResult with normalized English text
        """
        # Detect language
        language = self.detect_language(text)
        
        if language == 'en':
            # Pure English - minimal processing
            return ProcessingResult(
                original_text=text,
                normalized_text=text,
                detected_language='en',
                extracted_entities=[],
                phonetic_matches=[],
                confidence=1.0
            )
        
        # Process Hinglish/Hindi
        normalized_text = text.lower()
        extracted_entities = []
        phonetic_matches = []
        
        # Extract known Hinglish terms
        term_matches = self.extract_hinglish_terms(text)
        
        for original, english, confidence in term_matches:
            # Replace in normalized text
            normalized_text = normalized_text.replace(original, english)
            
            phonetic_matches.append((original, english, confidence))
            extracted_entities.append({
                'original': original,
                'normalized': english,
                'type': 'medical_term',
                'confidence': confidence
            })
        
        # Fuzzy match remaining unknown terms
        words = normalized_text.split()
        for word in words:
            if word not in self._term_mappings.values():  # Not already translated
                fuzzy_result = self.fuzzy_match_term(word)
                if fuzzy_result:
                    original, english, confidence = fuzzy_result
                    if confidence >= self.config.fuzzy_threshold:
                        normalized_text = normalized_text.replace(word, english)
                        phonetic_matches.append((word, english, confidence))
        
        # Clean up the normalized text
        normalized_text = self._clean_text(normalized_text)
        
        # Calculate overall confidence
        if phonetic_matches:
            avg_confidence = sum(m[2] for m in phonetic_matches) / len(phonetic_matches)
        else:
            avg_confidence = 0.5
        
        return ProcessingResult(
            original_text=text,
            normalized_text=normalized_text,
            detected_language=language,
            extracted_entities=extracted_entities,
            phonetic_matches=phonetic_matches,
            confidence=avg_confidence
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize processed text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common Hindi particles that don't translate
        particles = ['hai', 'ho', 'hain', 'tha', 'thi', 'the', 'raha', 'rahi', 
                    'rahe', 'me', 'mein', 'ka', 'ki', 'ke', 'ko', 'se', 'aur',
                    'ya', 'bhi', 'nahi', 'na', 'kya', 'kaise', 'kab', 'kahan']
        
        words = text.split()
        cleaned_words = [w for w in words if w not in particles or w in self._term_mappings.values()]
        
        return ' '.join(cleaned_words)
    
    async def translate_with_gemini(self, text: str, api_key: str) -> str:
        """
        Use Gemini AI for complex translations.
        
        Fallback for text that can't be handled by rule-based processing.
        """
        import urllib.request
        import json
        
        prompt = f"""Translate the following Hindi/Hinglish medical text to English.
Keep medical terms accurate. Only return the translation.

Text: {text}

Translation:"""
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            
            data = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100}
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            
            loop = asyncio.get_event_loop()
            
            def _request():
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.loads(response.read().decode('utf-8'))
            
            result = await loop.run_in_executor(self._executor, _request)
            
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            logger.warning(f"Gemini translation failed: {e}")
        
        return text
