from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import chromadb
import faiss
import numpy as np
from dataclasses import dataclass

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]], ids: List[str]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        pass

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "papers", persist_directory: str = None):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]], ids: List[str]) -> None:
        pass
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        pass
    
    def delete_documents(self, ids: List[str]) -> None:
        pass

class FAISSVectorStore(VectorStore):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = {}
        self.metadata = {}
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]], ids: List[str]) -> None:
        pass
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        pass
    
    def delete_documents(self, ids: List[str]) -> None:
        pass