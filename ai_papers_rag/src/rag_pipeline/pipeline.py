from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from ..retrieval.retriever import DocumentRetriever, RetrievalResult
from ..retrieval.query_processor import QueryProcessor, ProcessedQuery
from ..llm.llm_client import LLMClient, LLMResponse
from ..llm.prompt_templates import PromptTemplates

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    processing_time: float
    retrieval_results: RetrievalResult
    llm_response: LLMResponse

class RAGPipeline:
    def __init__(self, 
                 retriever: DocumentRetriever, 
                 llm_client: LLMClient,
                 query_processor: QueryProcessor = None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.query_processor = query_processor or QueryProcessor()
        self.prompt_templates = PromptTemplates()
    
    def query(self, user_query: str, k: int = 5, 
              include_sources: bool = True) -> RAGResponse:
        pass
    
    def _process_query(self, query: str) -> ProcessedQuery:
        pass
    
    def _retrieve_documents(self, processed_query: ProcessedQuery, k: int) -> RetrievalResult:
        pass
    
    def _generate_response(self, query: str, context_documents: List[str]) -> LLMResponse:
        pass
    
    def _format_sources(self, retrieval_results: RetrievalResult) -> List[Dict[str, Any]]:
        pass
    
    def _prepare_context(self, retrieval_results: RetrievalResult) -> List[str]:
        pass