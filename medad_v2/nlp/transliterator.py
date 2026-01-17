"""
MEDAD 2.0 - Hindi Transliterator
===================================

Bidirectional transliteration between Devanagari and Roman scripts.
"""

from typing import Optional
import re


# Devanagari to Roman mapping
DEVANAGARI_TO_ROMAN = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ri',
    
    # Vowel matras
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ृ': 'ri',
    
    # Consonants
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'व': 'w',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    
    # Special characters
    'ं': 'n', 'ः': 'h', '्': '', 'ँ': 'n',
    
    # Nukta consonants
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'फ़': 'f',
}


class Transliterator:
    """
    Bidirectional transliterator for Hindi text.
    
    Supports:
    - Devanagari → Roman (transliteration)
    - Roman → Devanagari (reverse transliteration)
    """
    
    def __init__(self):
        """Initialize the transliterator"""
        self._dev_to_roman = DEVANAGARI_TO_ROMAN.copy()
        self._roman_to_dev = {v: k for k, v in DEVANAGARI_TO_ROMAN.items()}
    
    def devanagari_to_roman(self, text: str) -> str:
        """
        Transliterate Devanagari text to Roman script.
        
        Args:
            text: Text in Devanagari script
        
        Returns:
            Romanized text
        """
        result = []
        
        for char in text:
            if char in self._dev_to_roman:
                result.append(self._dev_to_roman[char])
            elif char.isspace() or char.isascii():
                result.append(char)
            else:
                result.append(char)  # Keep unknown characters
        
        return ''.join(result)
    
    def roman_to_devanagari(self, text: str) -> str:
        """
        Transliterate Roman text to Devanagari script.
        
        Note: This is approximate as Roman script loses some information.
        
        Args:
            text: Text in Roman script
        
        Returns:
            Text in Devanagari script
        """
        result = []
        i = 0
        
        while i < len(text):
            # Try multi-character matches first
            matched = False
            for length in range(3, 0, -1):
                chunk = text[i:i+length].lower()
                if chunk in self._roman_to_dev:
                    result.append(self._roman_to_dev[chunk])
                    i += length
                    matched = True
                    break
            
            if not matched:
                result.append(text[i])
                i += 1
        
        return ''.join(result)
    
    def is_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari characters"""
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(devanagari_pattern.search(text))
    
    def detect_and_transliterate(self, text: str) -> str:
        """
        Auto-detect script and transliterate to Roman.
        
        Args:
            text: Input text in any script
        
        Returns:
            Romanized text
        """
        if self.is_devanagari(text):
            return self.devanagari_to_roman(text)
        return text
