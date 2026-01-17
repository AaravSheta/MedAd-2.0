"""
MEDAD 2.0 - Medical Text Chunker
===================================

Intelligent text chunking for medical documents.
Preserves semantic coherence for better retrieval.
"""

from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger("MedicalTextChunker")


class MedicalTextChunker:
    """
    Chunker optimized for medical text.
    
    Preserves:
    - Drug names and dosages
    - Symptom descriptions
    - Treatment protocols
    - Medical entity references
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        respect_sentences: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            respect_sentences: Try to break at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentences = respect_sentences
        
        # Sentence boundary pattern
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        
        # Medical entity patterns to preserve
        self._preserve_patterns = [
            r'\d+\s*(?:mg|ml|g|mcg|iu|units?)',  # Dosages
            r'\d+\s*(?:times?|x)\s*(?:daily|per day|a day)',  # Frequency
            r'(?:every|each)\s*\d+\s*(?:hours?|days?|weeks?)',  # Intervals
        ]
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller segments.
        
        Args:
            text: Input text to chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        if self.respect_sentences:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_size(text)
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text respecting sentence boundaries"""
        sentences = self._sentence_pattern.split(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = ' '.join(current_chunk)[-self.chunk_overlap:]
                    current_chunk = [overlap_text, sentence]
                    current_size = len(overlap_text) + sentence_size
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_size(self, text: str) -> List[str]:
        """Simple size-based chunking"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                # Try to find a good break point
                break_point = text.rfind(' ', start, end)
                if break_point > start:
                    end = break_point
            
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_medicine_record(
        self,
        medicine: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from a medicine database record.
        
        Args:
            medicine: Medicine record dictionary
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        
        # Create main content
        name = medicine.get('name', medicine.get('Medicine Name', 'Unknown'))
        therapeutic_class = medicine.get('Therapeutic Class', 'General')
        uses = medicine.get('combined_use', '')
        
        main_content = f"""
        Medicine: {name}
        Therapeutic Class: {therapeutic_class}
        Uses: {uses}
        """
        
        # Chunk the content
        text_chunks = self.chunk_text(main_content)
        
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "content": chunk,
                "metadata": {
                    "name": name,
                    "therapeutic_class": therapeutic_class,
                    "chunk_index": i,
                    "type": "medicine"
                }
            })
        
        return chunks
