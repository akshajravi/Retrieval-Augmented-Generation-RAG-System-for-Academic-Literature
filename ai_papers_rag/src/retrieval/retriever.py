from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..embeddings.vector_store import VectorStore, SearchResult
from ..embeddings.embedding_service import EmbeddingService

@dataclass
class RetrievalResult:
    documents: List[SearchResult]
    query: str
    total_results: int
    retrieval_time: float

class DocumentRetriever:
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def retrieve(self, query: str, k: int = 5, 
                filter_metadata: Dict[str, Any] = None) -> RetrievalResult:
        pass
    
    def retrieve_with_reranking(self, query: str, k: int = 5, 
                              rerank_top_k: int = 20) -> RetrievalResult:
        pass
    
    def hybrid_search(self, query: str, k: int = 5, 
                     alpha: float = 0.5) -> RetrievalResult:
        pass
    
    def _semantic_search(self, query: str, k: int) -> List[SearchResult]:
        pass
    
    def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        pass