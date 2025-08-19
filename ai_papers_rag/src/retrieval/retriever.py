from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging
import re
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
        self.logger = logging.getLogger(__name__)
        
        # Simple stopwords for basic keyword filtering
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why', 'how'
        }
    
    def retrieve(self, query: str, k: int = 5, 
                filter_metadata: Dict[str, Any] = None) -> RetrievalResult:
        """Simple semantic retrieval - the main method for small collections."""
        start_time = time.time()
        
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            self.logger.info(f"Retrieving {k} documents for query: '{query[:50]}...'")
            
            # Get embedding for the query
            query_embedding = self.embedding_service.embed_text(query.strip())
            
            # Perform search with or without filters
            if filter_metadata:
                documents = self.vector_store.search_with_filter(
                    query_embedding, filter_metadata, k
                )
            else:
                documents = self.vector_store.search(query_embedding, k)
            
            retrieval_time = time.time() - start_time
            
            self.logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.3f}s")
            
            return RetrievalResult(
                documents=documents,
                query=query,
                total_results=len(documents),
                retrieval_time=retrieval_time
            )
            
        except ValueError as e:
            self.logger.error(f"Invalid query: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in retrieve: {str(e)}")
            retrieval_time = time.time() - start_time
            return RetrievalResult(
                documents=[],
                query=query,
                total_results=0,
                retrieval_time=retrieval_time
            )
    
    def keyword_search(self, query: str, k: int = 5) -> RetrievalResult:
        """Simple keyword search for small collections - just keyword counting."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Performing simple keyword search for: '{query[:50]}...'")
            
            # Extract keywords from query
            query_keywords = self._extract_keywords(query)
            
            if not query_keywords:
                return RetrievalResult(
                    documents=[],
                    query=query,
                    total_results=0,
                    retrieval_time=time.time() - start_time
                )
            
            # Get all documents (fine for 12 papers)
            all_documents = self._get_all_documents()
            
            # Score documents by keyword matches
            scored_docs = []
            for doc in all_documents:
                score = self._simple_keyword_score(query_keywords, doc.content)
                if score > 0:
                    scored_docs.append(SearchResult(
                        content=doc.content,
                        metadata=doc.metadata,
                        score=score,
                        chunk_id=doc.chunk_id
                    ))
            
            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x.score, reverse=True)
            final_docs = scored_docs[:k]
            
            retrieval_time = time.time() - start_time
            
            self.logger.info(f"Keyword search found {len(final_docs)} documents in {retrieval_time:.3f}s")
            
            return RetrievalResult(
                documents=final_docs,
                query=query,
                total_results=len(final_docs),
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}")
            retrieval_time = time.time() - start_time
            return RetrievalResult(
                documents=[],
                query=query,
                total_results=0,
                retrieval_time=retrieval_time
            )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query - simple approach."""
        # Convert to lowercase and extract words (including hyphenated terms)
        words = re.findall(r'\b\w+(?:[-_]\w+)*\b', query.lower())
        
        # Filter stopwords and short words
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in self.stopwords
        ]
        
        return keywords
    
    def _simple_keyword_score(self, query_keywords: List[str], document: str) -> float:
        """Simple keyword matching score - just count matches."""
        doc_lower = document.lower()
        score = 0.0
        
        for keyword in query_keywords:
            # Count exact word matches
            matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', doc_lower))
            score += matches
        
        # Normalize by query length for fair comparison
        return score / len(query_keywords) if query_keywords else 0.0
    
    def _get_all_documents(self) -> List[SearchResult]:
        """Get all documents - acceptable for small collections."""
        try:
            # For 12 papers, getting all documents is fine
            stats = self.vector_store.get_stats()
            total_docs = stats.get('total_documents', 0)
            
            if total_docs == 0:
                return []
            
            # Use a dummy embedding to get all documents
            dummy_embedding = [0.0] * self.embedding_service.get_embedding_dimension()
            all_docs = self.vector_store.search(dummy_embedding, k=total_docs)
            
            return all_docs
            
        except Exception as e:
            self.logger.error(f"Error getting all documents: {str(e)}")
            return []
    
    def search_by_paper(self, query: str, paper_name: str, k: int = 5) -> RetrievalResult:
        """Search within a specific paper - useful for small collections."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Searching in paper '{paper_name}' for: '{query[:50]}...'")
            
            # Create filter for specific paper
            filter_metadata = {"source_file": paper_name}
            
            # Use regular semantic search with filter
            query_embedding = self.embedding_service.embed_text(query.strip())
            documents = self.vector_store.search_with_filter(
                query_embedding, filter_metadata, k
            )
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                documents=documents,
                query=query,
                total_results=len(documents),
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            self.logger.error(f"Error searching in paper {paper_name}: {str(e)}")
            retrieval_time = time.time() - start_time
            return RetrievalResult(
                documents=[],
                query=query,
                total_results=0,
                retrieval_time=retrieval_time
            )
    
    def get_paper_list(self) -> List[str]:
        """Get list of all papers in the collection."""
        try:
            return self.vector_store.list_sources()
        except Exception as e:
            self.logger.error(f"Error getting paper list: {str(e)}")
            return []
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        try:
            vector_stats = self.vector_store.get_stats()
            embedding_dim = self.embedding_service.get_embedding_dimension()
            
            return {
                "total_documents": vector_stats.get('total_documents', 0),
                "total_papers": len(self.get_paper_list()),
                "embedding_dimension": embedding_dim,
                "vector_store_type": "ChromaDB"
            }
        except Exception as e:
            self.logger.error(f"Error getting retrieval stats: {str(e)}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Example of how to use the simplified retriever
    print("Simplified DocumentRetriever for small academic collections")
    print("\nMain features:")
    print("- retrieve(): Semantic search (primary method)")
    print("- keyword_search(): Simple keyword matching")
    print("- search_by_paper(): Search within specific paper")
    print("- get_paper_list(): List all papers")
    
    # Example usage:
    # retriever = DocumentRetriever(vector_store, embedding_service)
    
    # Main semantic search
    # result = retriever.retrieve("What is attention mechanism?", k=5)
    
    # Search within specific paper
    # result = retriever.search_by_paper("attention", "attention_is_all_you_need.pdf", k=3)
    
    # Simple keyword search
    # result = retriever.keyword_search("transformer architecture", k=5)
    
    # Get available papers
    # papers = retriever.get_paper_list()