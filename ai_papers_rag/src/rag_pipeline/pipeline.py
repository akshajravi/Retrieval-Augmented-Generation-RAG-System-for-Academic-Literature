from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging
from ..retrieval.retriever import DocumentRetriever, RetrievalResult
from ..retrieval.query_processor import QueryProcessor, ProcessedQuery
from ..llm.llm_client import LLMGenerator, AnswerResult  # Updated import
from ..llm.prompt_templates import PromptTemplates

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    processing_time: float
    retrieval_results: RetrievalResult
    llm_response: AnswerResult  # Updated to match your LLMGenerator
    confidence: str
    intent: str

class RAGPipeline:
    def __init__(self, 
                 retriever: DocumentRetriever, 
                 llm_generator: LLMGenerator,  # Updated parameter name
                 query_processor: QueryProcessor = None):
        self.retriever = retriever
        self.llm_generator = llm_generator
        self.query_processor = query_processor or QueryProcessor()
        self.logger = logging.getLogger(__name__)
    
    def query(self, user_query: str, k: int = 5, 
              include_sources: bool = True, temperature: float = 0.1) -> RAGResponse:
        """Main method to process a user query through the complete RAG pipeline."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: '{user_query[:50]}...'")
            
            # Step 1: Process and understand the query
            processed_query = self._process_query(user_query)
            self.logger.info(f"Query intent detected: {processed_query.intent}")
            
            # Step 2: Retrieve relevant documents
            retrieval_results = self._retrieve_documents(processed_query, k)
            self.logger.info(f"Retrieved {retrieval_results.total_results} documents")
            
            if not retrieval_results.documents:
                # No documents found, return empty response
                processing_time = time.time() - start_time
                return RAGResponse(
                    answer="I couldn't find any relevant information in the research papers to answer your question.",
                    sources=[],
                    query=user_query,
                    processing_time=processing_time,
                    retrieval_results=retrieval_results,
                    llm_response=None,
                    confidence="none",
                    intent=processed_query.intent
                )
            
            # Step 3: Generate response using LLM
            llm_response = self._generate_response(
                processed_query, 
                retrieval_results, 
                temperature
            )
            
            # Step 4: Format sources for response
            formatted_sources = self._format_sources(retrieval_results) if include_sources else []
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Query processed successfully in {processing_time:.3f}s")
            
            return RAGResponse(
                answer=llm_response.answer,
                sources=formatted_sources,
                query=user_query,
                processing_time=processing_time,
                retrieval_results=retrieval_results,
                llm_response=llm_response,
                confidence=llm_response.confidence,
                intent=processed_query.intent
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            processing_time = time.time() - start_time
            
            # Return error response
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                query=user_query,
                processing_time=processing_time,
                retrieval_results=RetrievalResult([], user_query, 0, 0.0),
                llm_response=None,
                confidence="none",
                intent="general"
            )
    
    def _process_query(self, query: str) -> ProcessedQuery:
        """Process and analyze the user query."""
        try:
            return self.query_processor.process_query(query)
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            # Return basic processed query as fallback
            return ProcessedQuery(
                original_query=query,
                cleaned_query=query.lower().strip(),
                keywords=[],
                intent="general",
                filters={}
            )
    
    def _retrieve_documents(self, processed_query: ProcessedQuery, k: int) -> RetrievalResult:
        """Retrieve relevant documents based on processed query."""
        try:
            # For now, disable complex filtering to avoid ChromaDB issues
            # Use simple semantic search
            return self.retriever.retrieve(processed_query.cleaned_query, k=k)
                
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            # Return empty result
            return RetrievalResult(
                documents=[],
                query=processed_query.cleaned_query,
                total_results=0,
                retrieval_time=0.0
            )
    
    def _generate_response(self, processed_query: ProcessedQuery, 
                          retrieval_results: RetrievalResult, 
                          temperature: float) -> AnswerResult:
        """Generate response using LLM with retrieved context."""
        try:
            return self.llm_generator.generate_answer_with_citations(
                query=processed_query.original_query,
                search_results=retrieval_results.documents,
                temperature=temperature,
                intent=processed_query.intent
            )
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {str(e)}")
            # Return error response
            return AnswerResult(
                answer=f"Error generating response: {str(e)}",
                sources=[],
                confidence="none",
                model_used="unknown",
                tokens_used=0,
                response_time=0.0
            )
    
    def _format_sources(self, retrieval_results: RetrievalResult) -> List[Dict[str, Any]]:
        """Format source information for the response."""
        formatted_sources = []
        
        for i, result in enumerate(retrieval_results.documents, 1):
            source_info = {
                "rank": i,
                "source_file": result.metadata.get('source_file', 'Unknown'),
                "source_name": PromptTemplates.clean_source_name(
                    result.metadata.get('source_file', 'Unknown')
                ),
                "similarity_score": round(result.score * 100, 2),  # Convert to percentage
                "chunk_id": result.chunk_id,
                "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "metadata": {
                    key: value for key, value in result.metadata.items() 
                    if key not in ['source_file']  # Don't duplicate source_file
                }
            }
            formatted_sources.append(source_info)
        
        return formatted_sources
    
    def _prepare_context(self, retrieval_results: RetrievalResult) -> List[str]:
        """Prepare context strings from retrieval results."""
        return [result.content for result in retrieval_results.documents]
    
    # Convenience methods for different types of queries
    
    def ask_definition(self, term: str, k: int = 3) -> RAGResponse:
        """Ask for a definition of a specific term."""
        query = f"What is {term}? Please provide a clear definition."
        return self.query(query, k=k)
    
    def ask_comparison(self, term1: str, term2: str, k: int = 5) -> RAGResponse:
        """Compare two concepts."""
        query = f"Compare {term1} and {term2}. What are the similarities and differences?"
        return self.query(query, k=k)
    
    def ask_methodology(self, method: str, k: int = 5) -> RAGResponse:
        """Ask about how something works or is implemented."""
        query = f"How does {method} work? Explain the methodology and approach."
        return self.query(query, k=k)
    
    def search_by_paper(self, paper_name: str, question: str, k: int = 3) -> RAGResponse:
        """Search within a specific paper."""
        query = f"In the paper {paper_name}, {question}"
        return self.query(query, k=k)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline components."""
        try:
            retrieval_stats = self.retriever.get_retrieval_stats()
            llm_stats = self.llm_generator.get_model_info()
            
            return {
                "retrieval_stats": retrieval_stats,
                "llm_stats": llm_stats,
                "available_intents": PromptTemplates.get_available_intents(),
                "pipeline_ready": True
            }
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {str(e)}")
            return {"error": str(e), "pipeline_ready": False}
    
    def get_available_papers(self) -> List[str]:
        """Get list of available papers in the system."""
        try:
            return self.retriever.get_paper_list()
        except Exception as e:
            self.logger.error(f"Error getting paper list: {str(e)}")
            return []
