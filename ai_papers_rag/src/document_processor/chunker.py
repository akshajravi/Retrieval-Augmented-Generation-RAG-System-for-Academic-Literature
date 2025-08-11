from typing import List, Dict, Any
from dataclasses import dataclass
import tiktoken
import re
import logging


@dataclass
class TextChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_index: int
    end_index: int

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        # Input validation
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []
        
        # Validate chunk size parameters
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        try:
            return self.chunk_by_sentences(text, metadata)
        except Exception as e:
            self.logger.error(f"Error in chunk_text: {e}")
            # Return empty list on error to prevent pipeline failure
            return []
    
    def chunk_by_tokens(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        # Input validation
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []

        if metadata is None:
            metadata = {}
        
        try:
            #tokenizer is encoding object that converts text to token
            tokens = self.tokenizer.encode(text)
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            return []
        
        # Log warning for very large documents
        if len(tokens) > 50000:  # ~50k tokens
            self.logger.warning(f"Processing large document with {len(tokens)} tokens")
        
        chunks = []

        step_size = self.chunk_size - self.chunk_overlap
        current_char_pos = 0
        
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i+self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Calculate character positions more efficiently
            if i == 0:
                start_char = 0
            else:
                # For subsequent chunks, decode only the step_size tokens to get position increment
                step_tokens = tokens[i-step_size:i]
                step_text = self.tokenizer.decode(step_tokens)
                current_char_pos += len(step_text)
                start_char = current_char_pos
            
            end_char = start_char + len(chunk_text)

            #appending new information to metadata

            chunk_metadata = {
                **metadata,
                "chunk_method": "token_based",
                "chunk_index": len(chunks),
                "token_count": len(chunk_tokens),
                "char_start": start_char,
                "char_end": end_char
            }

            chunk_id = f"{metadata.get('source_file', 'unknown')}_{len(chunks)}"

            # Only add valid chunks
            if self._is_valid_chunk(chunk_text):
                chunks.append(TextChunk(
                    content=chunk_text.strip(),
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    start_index=start_char,
                    end_index=end_char
                ))
        
        return chunks

    
    def chunk_by_sentences(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        # Input validation
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []
        
        if metadata is None:
            metadata = {}

        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_start = 0

        for paragraph in paragraphs:
            paragraph_text = paragraph["text"]
            paragraph_start = paragraph["start"]

            potential_chunk = current_chunk + "\n\n" + paragraph_text if current_chunk else paragraph_text
            potential_tokens= len(self.tokenizer.encode(potential_chunk))

            if potential_tokens <= self.chunk_size:

                if current_chunk:
                    current_chunk += "\n\n" + paragraph_text
                else:
                    current_chunk = paragraph_text
                    current_start = paragraph_start
            else:
                if current_chunk:
                    chunk = self._create_chunk_with_overlap(current_chunk, chunks, metadata, current_start)
                    chunks.append(chunk)
                
                if len(self.tokenizer.encode(paragraph_text)) > self.chunk_size:

                    sentence_chunks = self._split_large_paragraph(paragraph_text, paragraph_start, metadata)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph_text
                    current_start = paragraph_start

        if current_chunk:
            chunk = self._create_chunk_with_overlap(current_chunk, chunks, metadata, current_start)
            chunks.append(chunk)
        
        return chunks
    

    def _split_into_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        paragraphs = []
        
        # Use regex finditer to get both content and positions reliably
        paragraph_pattern = r'(.+?)(?:\n\s*\n|$)'
        current_pos = 0
        
        for match in re.finditer(paragraph_pattern, text, re.DOTALL):
            paragraph_text = match.group(1).strip()
            
            if paragraph_text:  # Skip empty paragraphs
                # Get actual start position of the cleaned paragraph text
                match_start = match.start(1)
                
                # Find where the stripped text actually starts within the match
                original_match_text = match.group(1)
                stripped_start_offset = len(original_match_text) - len(original_match_text.lstrip())
                actual_start = match_start + stripped_start_offset
                
                paragraphs.append({
                    "text": paragraph_text,
                    "start": actual_start,
                    "end": actual_start + len(paragraph_text)
                })
        
        return paragraphs
    
    def _find_protected_spans(self, text: str) -> List[Dict[str, Any]]:
        """Find spans that should not be split (citations, equations, etc.)"""
        protected_spans = []
        
        # LaTeX equations (display math)
        for match in re.finditer(r'\$\$.*?\$\$', text, re.DOTALL):
            protected_spans.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'equation_display',
                'content': match.group()
            })
        
        # LaTeX equations (inline math)
        for match in re.finditer(r'\$[^$]+\$', text):
            protected_spans.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'equation_inline',
                'content': match.group()
            })
        
        # Citations - numeric [1], [2-5], [1,3,5]
        for match in re.finditer(r'\[\d+(?:[-,]\d+)*\]', text):
            protected_spans.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'citation_numeric',
                'content': match.group()
            })
        
        # Citations - author-year (Smith et al., 2020)
        for match in re.finditer(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?\)', text):
            protected_spans.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'citation_author',
                'content': match.group()
            })
        
        # Section headers (## Header, 1. Header, etc.)
        for match in re.finditer(r'^(?:#{1,6}\s+.+|^\d+\.?\s+[A-Z].+)$', text, re.MULTILINE):
            protected_spans.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'section_header',
                'content': match.group()
            })
        
        # Sort by start position
        protected_spans.sort(key=lambda x: x['start'])
        return protected_spans
    
    def _should_avoid_split_at(self, text: str, position: int) -> bool:
        """Check if splitting at this position would break a protected pattern"""
        protected_spans = self._find_protected_spans(text)
        
        for span in protected_spans:
            # Don't split within protected spans
            if span['start'] < position < span['end']:
                return True
            # Don't split too close to equations or citations
            if span['type'] in ['equation_display', 'equation_inline', 'citation_numeric', 'citation_author']:
                if abs(position - span['start']) < 10 or abs(position - span['end']) < 10:
                    return True
        
        return False
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting that handles abbreviations properly"""
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations = {
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'B.S.', 'M.S.',
            'et al.', 'i.e.', 'e.g.', 'vs.', 'etc.', 'cf.', 'ca.', 'viz.', 'Inc.', 'Corp.', 'Ltd.',
            'Fig.', 'Table', 'Eq.', 'Ref.', 'Ch.', 'Sec.', 'Vol.', 'No.', 'pp.', 'p.',
            'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.'
        }
        
        # Create a pattern that matches sentence endings but not abbreviations
        # First, temporarily replace abbreviations
        temp_text = text
        abbrev_placeholders = {}
        
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            temp_text = temp_text.replace(abbrev, placeholder)
            abbrev_placeholders[placeholder] = abbrev
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', temp_text)
        
        # Restore abbreviations
        restored_sentences = []
        for sentence in sentences:
            restored_sentence = sentence
            for placeholder, original in abbrev_placeholders.items():
                restored_sentence = restored_sentence.replace(placeholder, original)
            restored_sentences.append(restored_sentence.strip())
        
        # Filter out empty sentences
        return [s for s in restored_sentences if s]
    
    def _is_valid_chunk(self, chunk_text: str) -> bool:
        """Validate chunk quality and filter out poor chunks"""
        if not chunk_text or not chunk_text.strip():
            return False
        
        # Check minimum length (avoid tiny chunks)
        if len(chunk_text.strip()) < 20:
            return False
        
        # Check if chunk has meaningful content (not just whitespace/punctuation)
        meaningful_chars = re.sub(r'[\s\n\t\r.,;:!?()\[\]{}\"\'-]', '', chunk_text)
        if len(meaningful_chars) < 10:
            return False
        
        # Check text-to-total character ratio
        total_chars = len(chunk_text)
        text_chars = len(re.sub(r'[^a-zA-Z0-9]', '', chunk_text))
        if text_chars / total_chars < 0.3:  # At least 30% meaningful characters
            return False
        
        return True
    
    def _create_chunk_with_overlap(self, chunk_text: str, existing_chunks: List[TextChunk], 
                                   metadata: Dict[str, Any], start_pos: int) -> TextChunk:
        
        final_chunk_text = chunk_text
        
        # Add overlap from previous chunk if it exists
        if existing_chunks and self.chunk_overlap > 0:
            prev_chunk = existing_chunks[-1]
            prev_tokens = self.tokenizer.encode(prev_chunk.content)
            
            if len(prev_tokens) >= self.chunk_overlap:
                overlap_tokens = prev_tokens[-self.chunk_overlap:]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                final_chunk_text = overlap_text + "\n\n" + chunk_text
        
        # Calculate token count and positions
        token_count = len(self.tokenizer.encode(final_chunk_text))
        end_pos = start_pos + len(chunk_text)
        
        chunk_metadata = {
            **metadata,
            "chunk_method": "semantic",
            "chunk_index": len(existing_chunks),
            "token_count": token_count,
            "has_overlap": len(existing_chunks) > 0 and self.chunk_overlap > 0
        }
        
        chunk_id = f"{metadata.get('source_file', 'unknown')}_{len(existing_chunks)}"
        
        return TextChunk(
            content=final_chunk_text.strip(),
            metadata=chunk_metadata,
            chunk_id=chunk_id,
            start_index=start_pos,
            end_index=end_pos
        )
    
    def _split_large_paragraph(self, paragraph: str, start_pos: int, 
                              metadata: Dict[str, Any]) -> List[TextChunk]:
        
        sentences = self._split_into_sentences(paragraph)
        chunks = []
        current_chunk = ""
        current_start = start_pos
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            potential_tokens = len(self.tokenizer.encode(potential_chunk))
            
            if potential_tokens <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunk = self._create_chunk_with_overlap(
                        current_chunk, chunks, metadata, current_start
                    )
                    chunks.append(chunk)
                
                current_chunk = sentence
                current_start = start_pos + paragraph.find(sentence)
        
        # Last chunk
        if current_chunk:
            chunk = self._create_chunk_with_overlap(
                current_chunk, chunks, metadata, current_start
            )
            chunks.append(chunk)
        
        return chunks