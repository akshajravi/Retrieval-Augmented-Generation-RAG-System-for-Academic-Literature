from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from ..retrieval.retriever import RetrievalResult

@dataclass
class FormattedResponse:
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    confidence_score: float

class ResponseFormatter:
    def __init__(self):
        # Pattern to match citations in various formats
        self.citation_patterns = [
            r'\[([^\]]+)\]',  # [Source Name] or [1] or [Author et al.]
            r'\(([^)]+)\)',   # (Source Name) format
        ]
        
        # Confidence indicators in text
        self.high_confidence_indicators = [
            'specifically', 'clearly states', 'demonstrates', 'shows that',
            'according to', 'as stated in', 'research indicates', 'study found'
        ]
        
        self.low_confidence_indicators = [
            'might', 'could', 'possibly', 'appears to', 'seems to',
            'suggests', 'may indicate', 'potentially', 'unclear'
        ]
    
    def format_response(self, 
                      answer: str, 
                      retrieval_results: RetrievalResult,
                      include_citations: bool = True) -> FormattedResponse:
        """Main method to format a complete response with sources and citations."""
        try:
            # Format sources from retrieval results
            formatted_sources = self.format_sources(retrieval_results)
            
            # Add citations to answer if requested
            if include_citations and formatted_sources:
                cited_answer = self.add_citations(answer, formatted_sources)
            else:
                cited_answer = answer
            
            # Extract citations from the answer
            citations = self._extract_citations(cited_answer)
            
            # Calculate confidence score
            confidence = self.calculate_confidence(cited_answer, formatted_sources)
            
            return FormattedResponse(
                answer=cited_answer,
                sources=formatted_sources,
                citations=citations,
                confidence_score=confidence
            )
            
        except Exception as e:
            # Return minimal response on error
            return FormattedResponse(
                answer=answer,
                sources=[],
                citations=[],
                confidence_score=0.0
            )
    
    def add_citations(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Add numbered citations to answer based on available sources."""
        if not sources:
            return answer
        
        cited_answer = answer
        
        # Create mapping of source names to citation numbers
        source_map = {}
        for i, source in enumerate(sources, 1):
            source_name = source.get('source_name', f'Source {i}')
            source_map[source_name.lower()] = str(i)
        
        # Replace existing citations with numbered format
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, cited_answer)
            for match in reversed(list(matches)):  # Reverse to maintain positions
                citation_text = match.group(1).lower()
                
                # Try to find matching source
                citation_number = None
                for source_name, number in source_map.items():
                    if source_name in citation_text or citation_text in source_name:
                        citation_number = number
                        break
                
                if citation_number:
                    # Replace with numbered citation
                    start, end = match.span()
                    cited_answer = (cited_answer[:start] + 
                                  f"[{citation_number}]" + 
                                  cited_answer[end:])
        
        return cited_answer
    
    def format_sources(self, retrieval_results: RetrievalResult) -> List[Dict[str, Any]]:
        """Format source information from retrieval results."""
        formatted_sources = []
        
        for i, result in enumerate(retrieval_results.documents, 1):
            # Extract source file information
            source_file = result.metadata.get('source_file', 'Unknown Source')
            source_name = self._clean_source_name(source_file)
            
            # Create formatted source entry
            source_info = {
                "citation_number": i,
                "source_name": source_name,
                "source_file": source_file,
                "similarity_score": round(result.score, 3),
                "relevance": self._score_to_relevance(result.score),
                "content_preview": self._create_content_preview(result.content),
                "metadata": {
                    "chunk_id": result.chunk_id,
                    "token_count": result.metadata.get('token_count', 0),
                    "chunk_index": result.metadata.get('chunk_index', 0)
                }
            }
            
            # Add additional metadata if available
            if 'authors' in result.metadata:
                source_info['authors'] = result.metadata['authors']
            if 'publication_date' in result.metadata:
                source_info['publication_date'] = result.metadata['publication_date']
            
            formatted_sources.append(source_info)
        
        return formatted_sources
    
    def calculate_confidence(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on answer content and source quality."""
        if not sources or not answer.strip():
            return 0.0
        
        confidence_score = 0.0
        
        # Factor 1: Source quality (40% weight)
        if sources:
            avg_relevance = sum(source.get('similarity_score', 0) for source in sources) / len(sources)
            source_quality = min(avg_relevance * 0.4, 0.4)
            confidence_score += source_quality
        
        # Factor 2: Number of sources (20% weight)
        source_count_score = min(len(sources) / 5.0, 1.0) * 0.2  # Max score with 5+ sources
        confidence_score += source_count_score
        
        # Factor 3: Answer specificity (25% weight)
        specificity_score = self._calculate_specificity_score(answer) * 0.25
        confidence_score += specificity_score
        
        # Factor 4: Citation presence (15% weight)
        citations = self._extract_citations(answer)
        citation_score = min(len(citations) / len(sources), 1.0) * 0.15 if sources else 0
        confidence_score += citation_score
        
        return round(min(confidence_score, 1.0), 3)
    
    def format_for_display(self, formatted_response: FormattedResponse) -> str:
        """Format response for human-readable display."""
        display_parts = []
        
        # Add main answer
        display_parts.append("## Answer")
        display_parts.append(formatted_response.answer)
        display_parts.append("")
        
        # Add confidence indicator
        confidence_text = self._confidence_to_text(formatted_response.confidence_score)
        display_parts.append(f"**Confidence:** {confidence_text} ({formatted_response.confidence_score:.1%})")
        display_parts.append("")
        
        # Add sources if available
        if formatted_response.sources:
            display_parts.append("## Sources")
            for source in formatted_response.sources:
                source_line = f"[{source['citation_number']}] **{source['source_name']}**"
                if 'authors' in source:
                    source_line += f" - {source['authors']}"
                if 'publication_date' in source:
                    source_line += f" ({source['publication_date']})"
                
                source_line += f" - Relevance: {source['relevance']}"
                display_parts.append(source_line)
                
                # Add content preview
                if source.get('content_preview'):
                    display_parts.append(f"   *Preview: {source['content_preview']}*")
                display_parts.append("")
        
        return "\n".join(display_parts)
    
    def format_for_api(self, formatted_response: FormattedResponse) -> Dict[str, Any]:
        """Format response for API consumption."""
        return {
            "answer": formatted_response.answer,
            "confidence": {
                "score": formatted_response.confidence_score,
                "level": self._confidence_to_text(formatted_response.confidence_score),
                "percentage": f"{formatted_response.confidence_score:.1%}"
            },
            "sources": formatted_response.sources,
            "citations": formatted_response.citations,
            "metadata": {
                "source_count": len(formatted_response.sources),
                "citation_count": len(formatted_response.citations),
                "has_high_confidence": formatted_response.confidence_score >= 0.7
            }
        }
    
    # Helper methods
    
    def _clean_source_name(self, source_file: str) -> str:
        """Convert filename to readable source name."""
        name = source_file.replace('.pdf', '')
        name = name.replace('_', ' ').title()
        return name
    
    def _score_to_relevance(self, score: float) -> str:
        """Convert similarity score to human-readable relevance."""
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _create_content_preview(self, content: str, max_length: int = 150) -> str:
        """Create a preview of the content."""
        if len(content) <= max_length:
            return content.strip()
        
        # Try to cut at sentence boundary
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.7:  # If period is in last 30%
            return truncated[:last_period + 1]
        else:
            return truncated.strip() + "..."
    
    def _extract_citations(self, answer: str) -> List[str]:
        """Extract all citations from the answer."""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, answer)
            citations.extend(matches)
        
        # Remove duplicates while preserving order
        unique_citations = []
        for citation in citations:
            if citation not in unique_citations:
                unique_citations.append(citation)
        
        return unique_citations
    
    def _calculate_specificity_score(self, answer: str) -> float:
        """Calculate how specific/detailed the answer is."""
        answer_lower = answer.lower()
        
        # Count high confidence indicators
        high_indicators = sum(1 for indicator in self.high_confidence_indicators 
                            if indicator in answer_lower)
        
        # Count low confidence indicators (negative impact)
        low_indicators = sum(1 for indicator in self.low_confidence_indicators 
                           if indicator in answer_lower)
        
        # Base score from answer length and detail
        length_score = min(len(answer) / 500.0, 1.0)  # Normalize to 0-1
        
        # Adjust based on indicators
        indicator_adjustment = (high_indicators * 0.1) - (low_indicators * 0.1)
        
        return max(0.0, min(1.0, length_score + indicator_adjustment))
    
    def _confidence_to_text(self, confidence_score: float) -> str:
        """Convert confidence score to human-readable text."""
        if confidence_score >= 0.8:
            return "Very High"
        elif confidence_score >= 0.6:
            return "High"
        elif confidence_score >= 0.4:
            return "Medium"
        elif confidence_score >= 0.2:
            return "Low"
        else:
            return "Very Low"

