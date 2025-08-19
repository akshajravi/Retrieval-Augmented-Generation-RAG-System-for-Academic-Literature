from typing import List, Dict, Any, Optional
from openai import OpenAI
from dataclasses import dataclass
import logging
import time
import re
import tiktoken
from .prompt_templates import PromptTemplates

@dataclass
class LLMResponse:
    content: str
    model_used: str
    tokens_used: int
    finish_reason: str
    response_time: float

@dataclass
class AnswerResult:
    answer: str
    sources: List[str]
    confidence: str
    model_used: str
    tokens_used: int
    response_time: float

class LLMGenerator:
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
        # Get tokenizer for precise token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Use centralized prompt templates
        self.prompt_templates = PromptTemplates()
    
    def generate_answer_with_citations(self, query: str, search_results: List[Any], 
                                     temperature: float = 0.1, intent: str = "general") -> AnswerResult:
        """Generate a comprehensive answer with proper citations from search results."""
        start_time = time.time()
        
        try:
            if not search_results:
                return AnswerResult(
                    answer="I couldn't find any relevant information in the research papers to answer your question.",
                    sources=[],
                    confidence="none",
                    model_used=self.model,
                    tokens_used=0,
                    response_time=time.time() - start_time
                )
            
            # Use PromptTemplates to format prompt with intent
            prompt = self.prompt_templates.format_rag_prompt_with_search_results(
                self._sanitize_input(query), 
                search_results, 
                intent
            )
            
            # Extract source mapping for citation parsing
            source_mapping = self._get_source_mapping(search_results)
            
            # Calculate token limits properly
            prompt_tokens = len(self.tokenizer.encode(prompt))
            max_response_tokens = self._calculate_max_response_tokens(prompt_tokens)
            
            if max_response_tokens < 50:
                # Context too long, need to truncate and regenerate prompt
                truncated_results = self._truncate_search_results(search_results, query)
                prompt = self.prompt_templates.format_rag_prompt_with_search_results(
                    self._sanitize_input(query), 
                    truncated_results, 
                    intent
                )
                source_mapping = self._get_source_mapping(truncated_results)
                prompt_tokens = len(self.tokenizer.encode(prompt))
                max_response_tokens = self._calculate_max_response_tokens(prompt_tokens)
            
            # Generate response
            llm_response = self._call_openai(prompt, temperature, max_response_tokens)
            
            # Parse response and extract citations
            parsed_answer = self._parse_citations(llm_response.content, source_mapping)
            
            # Assess confidence based on search quality
            confidence = self._assess_confidence(search_results, parsed_answer)
            
            return AnswerResult(
                answer=parsed_answer["answer"],
                sources=parsed_answer["sources"],
                confidence=confidence,
                model_used=llm_response.model_used,
                tokens_used=llm_response.tokens_used,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error generating answer with citations: {str(e)}")
            return AnswerResult(
                answer=f"Error generating answer: {str(e)}",
                sources=[],
                confidence="none",
                model_used=self.model,
                tokens_used=0,
                response_time=time.time() - start_time
            )
    
    def _get_source_mapping(self, search_results: List[Any]) -> Dict[str, str]:
        """Extract source mapping from search results for citation parsing."""
        source_mapping = {}
        
        for i, result in enumerate(search_results, 1):
            source_file = result.metadata.get('source_file', 'Unknown Source')
            source_name = PromptTemplates.clean_source_name(source_file)
            source_id = f"SOURCE_{i}"
            source_mapping[source_id] = source_name
        
        return source_mapping
    
    
    def _sanitize_input(self, text: str) -> str:
        """Basic sanitization to prevent prompt injection."""
        # Remove or escape potential prompt injection patterns
        sanitized = text.replace('"""', '"').replace("'''", "'")
        # Remove system/assistant role indicators
        sanitized = re.sub(r'\b(system|assistant|user):\s*', '', sanitized, flags=re.IGNORECASE)
        return sanitized.strip()
    
    def _calculate_max_response_tokens(self, prompt_tokens: int) -> int:
        """Calculate maximum tokens available for response."""
        # Get model's context limit
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }
        
        context_limit = model_limits.get(self.model, 4096)
        
        # Reserve tokens for response (at least 200, up to 1500)
        safety_margin = 50  # Small safety margin
        max_response = min(1500, max(200, context_limit - prompt_tokens - safety_margin))
        
        return max_response
    
    def _truncate_search_results(self, search_results: List[Any], query: str) -> List[Any]:
        """Truncate search results to fit within token limits."""
        # Calculate available tokens
        base_prompt = self.prompt_templates.format_rag_prompt_with_search_results(query, [], "general")
        base_tokens = len(self.tokenizer.encode(base_prompt))
        available_for_context = 3000 - base_tokens  # Leave room for response
        
        truncated_results = []
        current_tokens = 0
        
        for result in search_results:
            # Estimate tokens for this result
            result_text = f"[SOURCE_{len(truncated_results)+1}]: {result.content}"
            result_tokens = len(self.tokenizer.encode(result_text))
            
            if current_tokens + result_tokens <= available_for_context:
                truncated_results.append(result)
                current_tokens += result_tokens
            else:
                break
        
        return truncated_results if truncated_results else search_results[:1]  # At least one result
    
    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Make the actual OpenAI API call."""
        start_time = time.time()
        
        try:
            # Use chat messages format with system prompt
            messages = [
                {"role": "system", "content": self.prompt_templates.RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model_used=self.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _parse_citations(self, response: str, source_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Parse citations with consistent format - only look for [SOURCE_X] patterns."""
        sources_cited = []
        
        # Only look for SOURCE_X patterns to avoid confusion
        for source_id, source_name in source_mapping.items():
            pattern = rf'\[{re.escape(source_id)}\]'
            if re.search(pattern, response):
                sources_cited.append(source_name)
                # Replace SOURCE_X with readable citation
                response = re.sub(pattern, f'[{source_name}]', response)
        
        # Clean up the answer
        cleaned_answer = self._clean_answer(response)
        
        # Remove duplicates while preserving order
        unique_sources = []
        for source in sources_cited:
            if source not in unique_sources:
                unique_sources.append(source)
        
        return {
            "answer": cleaned_answer,
            "sources": unique_sources
        }
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer."""
        # Remove extra whitespace
        answer = re.sub(r'\n\s*\n', '\n\n', answer)
        answer = answer.strip()
        
        # Ensure proper sentence endings
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer
    
    def _assess_confidence(self, search_results: List[Any], parsed_answer: Dict[str, Any]) -> str:
        """Improved confidence assessment based on search result quality."""
        if not search_results:
            return "none"
        
        # Factor 1: Quality of top search results (more important than quantity)
        top_scores = [result.score for result in search_results[:3]]  # Top 3 results
        avg_top_score = sum(top_scores) / len(top_scores)
        
        # Factor 2: Coverage - how many of the top results were cited
        cited_sources = set(parsed_answer["sources"])
        top_source_names = {
            PromptTemplates.clean_source_name(result.metadata.get('source_file', ''))
            for result in search_results[:3]
        }
        coverage = len(cited_sources.intersection(top_source_names)) / min(3, len(search_results))
        
        # Factor 3: Answer completeness (not just length)
        answer_has_specifics = any(
            indicator in parsed_answer["answer"].lower()
            for indicator in ['specifically', 'shows that', 'demonstrates', 'found that', 'according to']
        )
        
        # Calculate confidence
        score = 0
        
        # High quality search results
        if avg_top_score >= 0.8:
            score += 2
        elif avg_top_score >= 0.6:
            score += 1
        
        # Good coverage of top results
        if coverage >= 0.67:  # 2/3 of top results cited
            score += 2
        elif coverage >= 0.33:  # 1/3 of top results cited
            score += 1
        
        # Answer has specific information
        if answer_has_specifics:
            score += 1
        
        # Answer length (less important)
        if len(parsed_answer["answer"]) >= 100:
            score += 1
        
        # Map score to confidence level
        if score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        elif score >= 1:
            return "low"
        else:
            return "none"
    
    def estimate_tokens(self, text: str) -> int:
        """Accurate token estimation using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "tokenizer": self.tokenizer.name,
            "encoding": "tiktoken"
        }

