from typing import List, Dict, Any, Optional
from openai import OpenAI
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    model_used: str
    tokens_used: int
    finish_reason: str

class LLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.1, 
                         max_tokens: int = 2000) -> LLMResponse:
        pass
    
    @abstractmethod
    def generate_with_context(self, query: str, context: List[str], 
                            temperature: float = 0.1) -> LLMResponse:
        pass

class OpenAILLMClient(LLMClient):
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, prompt: str, temperature: float = 0.1, 
                         max_tokens: int = 2000) -> LLMResponse:
        pass
    
    def generate_with_context(self, query: str, context: List[str], 
                            temperature: float = 0.1) -> LLMResponse:
        pass
    
    def _prepare_messages(self, query: str, context: List[str]) -> List[Dict[str, str]]:
        pass