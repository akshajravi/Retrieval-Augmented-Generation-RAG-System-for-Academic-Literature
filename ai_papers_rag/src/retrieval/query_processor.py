from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class ProcessedQuery:
    original_query: str
    cleaned_query: str
    keywords: List[str]
    intent: str
    filters: Dict[str, Any]
    is_complex: bool = False

class QueryProcessor:
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        }
        
        
        self.question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which'}
        
        # Academic paper specific terms
        self.academic_terms = {
            'paper', 'study', 'research', 'article', 'publication', 'work',
            'author', 'authors', 'findings', 'results', 'conclusion', 'abstract',
            'methodology', 'approach', 'framework', 'model', 'algorithm'
        }
        
        self.intent_patterns = {
            'definition': {
                'patterns': [r'what is', r'define', r'definition of', r'meaning of', 
                           r'explain', r'describe', r'tell me about'],
                'score': 1
            },
            'comparison': {
                'patterns': [r'compare', r'difference between', r'vs\.?', r'versus', 
                           r'contrast', r'how.*different', r'similarities', r'differ'],
                'score': 2
            },
            'methodology': {
                'patterns': [r'how.*work', r'algorithm', r'method', r'approach', 
                           r'technique', r'implementation', r'process'],
                'score': 1
            },
            'findings': {
                'patterns': [r'results', r'findings', r'conclusion', r'outcome', 
                           r'performance', r'accuracy', r'evaluation'],
                'score': 1
            },
            'summary': {
                'patterns': [r'summarize', r'summary', r'overview', r'main points',
                           r'key findings', r'gist'],
                'score': 2
            },
            'specific_paper': {
                'patterns': [r'in.*paper', r'according to', r'from.*study', 
                           r'author.*says', r'paper.*mentions'],
                'score': 1
            }
        }
        
        self.filter_patterns = {
            'author': [
                r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'author[s]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s+(?:paper|study|work)',
                r'([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)',  # Multiple authors
                r'([A-Z][a-z]+)\s+et\s+al\.?'
            ],
            'year': [
                r'in\s+(\d{4})', r'from\s+(\d{4})', r'(\d{4})\s+paper',
                r'published\s+in\s+(\d{4})', r'(\d{4})\s+study'
            ],
            'concept': [
                r'about\s+([a-z\s]{3,})', r'on\s+([a-z\s]{3,})', 
                r'regarding\s+([a-z\s]{3,})'
            ]
        }
    
    def process_query(self, query: str) -> ProcessedQuery:
        """Main method to process a user query into structured format."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        cleaned = self.clean_query(query)
        
        keywords = self.extract_keywords(query)
        
        expanded_keywords = self._get_expanded_keywords(keywords)
        
        intent = self.detect_intent(query.lower())
        
        filters = self.extract_filters(query)
        
        is_complex = self.is_complex_query(query, keywords, intent, filters)
        
        return ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned,
            keywords=list(set(keywords + expanded_keywords)),  # Combine and deduplicate
            intent=intent,
            filters=filters,
            is_complex=is_complex
        )
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize the query text with less aggressive filtering."""
        cleaned = query.lower().strip()
        
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # More conservative character removal - keep more punctuation

        cleaned = re.sub(r'[^\w\s\-\(\)\.\']', '', cleaned)
        
        words = cleaned.split()
        filtered_words = []
        
        for i, word in enumerate(words):
            if i == 0 and word in self.question_words:
                filtered_words.append(word)
            elif word in self.academic_terms:
                filtered_words.append(word)
            elif word not in self.stopwords:
                filtered_words.append(word)
            elif self._is_part_of_important_phrase(word, words, i):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the original query."""
        words = re.findall(r'\b\w+(?:[-_]\w+)*\b', query.lower())
        
        keywords = []
        for word in words:
            if (len(word) > 2 and 
                word not in self.stopwords and 
                word not in self.question_words and
                (word.replace('-', '').replace('_', '').isalnum())):
                keywords.append(word)
        
        for word in words:
            if word in self.academic_terms and word not in keywords:
                keywords.append(word)
        
        return sorted(set(keywords), key=lambda x: (len(x), x in self.academic_terms), reverse=True)
    
    def detect_intent(self, query: str) -> str:
        """Detect intent using scoring system."""
        intent_scores = {}
        query_lower = query.lower()
        
        for intent, config in self.intent_patterns.items():
            score = 0
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, query_lower))
                score += matches * config['score']
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        if query_lower.startswith(('what', 'define')):
            return 'definition'
        elif query_lower.startswith('how'):
            return 'methodology'
        elif query_lower.startswith('why'): 
            return 'explanation'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return 'comparison'
        
        return 'general'
    
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from the query - DISABLED for now to avoid ChromaDB issues."""
        # Temporarily return empty filters to avoid ChromaDB filtering errors
        return {}
    
    def _get_expanded_keywords(self, keywords: List[str]) -> List[str]:
        """Get expanded synonyms for keywords."""
        synonyms = {
            'neural': ['network', 'net', 'ann'],
            'machine': ['learning', 'ml', 'ai', 'artificial'],
            'deep': ['learning', 'neural', 'dnn'],
            'attention': ['mechanism', 'self-attention', 'multi-head'],
            'transformer': ['model', 'architecture'],
            'bert': ['bidirectional', 'encoder'],
            'gpt': ['generative', 'transformer'],
            'nlp': ['natural', 'language', 'processing'],
            'algorithm': ['method', 'approach', 'technique']
        }
        
        expanded = []
        for keyword in keywords:
            if keyword in synonyms:
                expanded.extend(synonyms[keyword])
        
        return list(set(expanded))
    
    def _is_part_of_important_phrase(self, word: str, words: List[str], index: int) -> bool:
        """Check if a stopword is part of an important phrase."""
        important_phrases = [
            ['et', 'al'], ['state', 'of', 'the', 'art'], 
            ['end', 'to', 'end'], ['how', 'to']
        ]
        
        for phrase in important_phrases:
            if word in phrase:
                phrase_len = len(phrase)
                word_pos = phrase.index(word)
                start_idx = index - word_pos
                
                if (start_idx >= 0 and 
                    start_idx + phrase_len <= len(words) and
                    words[start_idx:start_idx + phrase_len] == phrase):
                    return True
        
        return False
    
    def is_complex_query(self, query: str, keywords: List[str], 
                        intent: str, filters: Dict[str, Any]) -> bool:
        """Determine if this is a complex query."""
        complexity_indicators = [
            len(keywords) > 5,
            intent in ['comparison', 'summary'],
            len(filters) > 1,
            bool(re.search(r'\b(and|or|but|however|also)\b', query.lower())),
            len(re.findall(r'\b(what|how|why|when|where)\b', query.lower())) > 1,
            len(query.split()) > 10
        ]
        
        return sum(complexity_indicators) >= 2