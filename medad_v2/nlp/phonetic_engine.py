"""
MEDAD 2.0 - Phonetic Matching Engine
=======================================

Phonetic algorithm implementations for matching Romanized Hindi variants.

Supports:
- Hindi Soundex (adapted for Devanagari phonetics)
- Double Metaphone for Hindi
- Fuzzy phonetic matching
"""

from typing import List, Tuple, Optional
import re


class PhoneticEngine:
    """
    Phonetic matching engine for Hindi/Hinglish text.
    
    Handles phonetic variations in Romanized Hindi:
    - Vowel length: 'a' vs 'aa', 'i' vs 'ee'
    - Aspirated consonants: 'k' vs 'kh', 'g' vs 'gh'
    - Retroflex vs dental: 'd' vs 'ḍ', 't' vs 'ṭ'
    """
    
    # Hindi Soundex mapping
    HINDI_SOUNDEX = {
        # Vowels - all map to 0
        'a': '0', 'e': '0', 'i': '0', 'o': '0', 'u': '0',
        
        # Labials - map to 1
        'b': '1', 'bh': '1', 'p': '1', 'ph': '1', 'f': '1', 'm': '1',
        
        # Dentals/Alveolars - map to 2
        't': '2', 'th': '2', 'd': '2', 'dh': '2', 'n': '2',
        
        # Retroflexes - map to 3
        'ṭ': '3', 'ṭh': '3', 'ḍ': '3', 'ḍh': '3', 'ṇ': '3',
        
        # Palatals - map to 4
        'c': '4', 'ch': '4', 'j': '4', 'jh': '4', 'ñ': '4',
        's': '4', 'sh': '4', 'z': '4',
        
        # Velars - map to 5
        'k': '5', 'kh': '5', 'g': '5', 'gh': '5', 'ṅ': '5',
        'q': '5', 'x': '5',
        
        # Semi-vowels and glides - map to 6
        'y': '6', 'r': '6', 'l': '6', 'v': '6', 'w': '6',
        
        # Fricatives - map to 7
        'h': '7',
    }
    
    # Common phonetic variations in Romanized Hindi
    PHONETIC_VARIATIONS = [
        # Vowel length variations
        ('aa', 'a'), ('ee', 'i'), ('ii', 'i'), ('oo', 'u'), ('uu', 'u'),
        
        # Aspirated/unaspirated
        ('kh', 'k'), ('gh', 'g'), ('ch', 'c'), ('jh', 'j'),
        ('th', 't'), ('dh', 'd'), ('ph', 'p'), ('bh', 'b'),
        
        # Retroflex/dental (often not distinguished in romanization)
        ('tt', 't'), ('dd', 'd'),
        
        # Sibilant variations
        ('sh', 's'), ('shh', 's'),
        
        # Common substitutions
        ('v', 'w'), ('f', 'ph'), ('z', 'j'),
    ]
    
    def __init__(self):
        """Initialize the phonetic engine"""
        self._variation_map = dict(self.PHONETIC_VARIATIONS)
    
    def hindi_soundex(self, word: str, length: int = 4) -> str:
        """
        Generate Hindi Soundex code for a word.
        
        Similar to American Soundex but adapted for Hindi phonetics.
        
        Args:
            word: Input word
            length: Length of soundex code (default 4)
        
        Returns:
            Soundex code string
        """
        if not word:
            return ''
        
        word = word.lower().strip()
        
        # Keep first letter
        code = word[0].upper()
        
        # Encode remaining letters
        prev_code = ''
        i = 1
        
        while i < len(word) and len(code) < length:
            # Check for digraphs first
            if i < len(word) - 1:
                digraph = word[i:i+2]
                if digraph in self.HINDI_SOUNDEX:
                    curr_code = self.HINDI_SOUNDEX[digraph]
                    if curr_code != '0' and curr_code != prev_code:
                        code += curr_code
                    prev_code = curr_code
                    i += 2
                    continue
            
            # Single character
            char = word[i]
            if char in self.HINDI_SOUNDEX:
                curr_code = self.HINDI_SOUNDEX[char]
                if curr_code != '0' and curr_code != prev_code:
                    code += curr_code
                prev_code = curr_code
            
            i += 1
        
        # Pad with zeros
        code = code.ljust(length, '0')
        
        return code[:length]
    
    def normalize_phonetic(self, word: str) -> str:
        """
        Normalize phonetic variations to a canonical form.
        
        Args:
            word: Input word with possible phonetic variations
        
        Returns:
            Normalized word
        """
        normalized = word.lower()
        
        # Remove doubled vowels (except at word boundaries)
        normalized = re.sub(r'([aeiou])\1+', r'\1', normalized)
        
        # Apply variation mappings (longer patterns first)
        for variant, standard in sorted(self.PHONETIC_VARIATIONS, key=lambda x: len(x[0]), reverse=True):
            normalized = normalized.replace(variant, standard)
        
        return normalized
    
    def phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate phonetic similarity between two words.
        
        Uses a combination of:
        1. Hindi Soundex matching
        2. Normalized form comparison
        3. Edit distance on phonetic representation
        
        Args:
            word1: First word
            word2: Second word
        
        Returns:
            Similarity score between 0 and 1
        """
        from difflib import SequenceMatcher
        
        # Soundex match
        soundex1 = self.hindi_soundex(word1)
        soundex2 = self.hindi_soundex(word2)
        soundex_match = 1.0 if soundex1 == soundex2 else 0.0
        
        # Normalized form match
        norm1 = self.normalize_phonetic(word1)
        norm2 = self.normalize_phonetic(word2)
        norm_ratio = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Combine scores (weighted average)
        combined = 0.4 * soundex_match + 0.6 * norm_ratio
        
        return combined
    
    def find_matches(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find phonetically similar matches from candidates.
        
        Args:
            query: Query word
            candidates: List of candidate words
            threshold: Minimum similarity threshold
        
        Returns:
            List of (candidate, similarity) tuples, sorted by similarity
        """
        matches = []
        
        for candidate in candidates:
            similarity = self.phonetic_similarity(query, candidate)
            if similarity >= threshold:
                matches.append((candidate, similarity))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def generate_variants(self, word: str) -> List[str]:
        """
        Generate phonetic variants of a word.
        
        Useful for fuzzy searching in databases.
        
        Args:
            word: Input word
        
        Returns:
            List of possible phonetic variants
        """
        variants = {word.lower()}
        
        # Generate variants by applying transformations
        for standard, variant in self.PHONETIC_VARIATIONS:
            for v in list(variants):
                if standard in v:
                    variants.add(v.replace(standard, variant))
                if variant in v:
                    variants.add(v.replace(variant, standard))
        
        return list(variants)
