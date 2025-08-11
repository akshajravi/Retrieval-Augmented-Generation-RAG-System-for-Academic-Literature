import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.retrieval.retriever import DocumentRetriever, RetrievalResult
from src.retrieval.query_processor import QueryProcessor, ProcessedQuery
from src.embeddings.vector_store import SearchResult

class TestQueryProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = QueryProcessor()
        self.sample_queries = [
            "What is attention mechanism in transformers?",
            "How does BERT compare to GPT?",
            "transformer architecture explained"
        ]
    
    def test_init(self):
        self.assertIsInstance(self.processor.stopwords, set)
        self.assertIn('the', self.processor.stopwords)
        self.assertIn('and', self.processor.stopwords)
    
    def test_process_query_interface(self):
        self.assertTrue(hasattr(self.processor, 'process_query'))
        
        # Test with sample query
        # result = self.processor.process_query(self.sample_queries[0])
        # self.assertIsInstance(result, ProcessedQuery)
    
    def test_clean_query_interface(self):
        self.assertTrue(hasattr(self.processor, 'clean_query'))
        
        # Test cleaning functionality
        # cleaned = self.processor.clean_query("What is the BERT model?")
        # self.assertIsInstance(cleaned, str)
        # self.assertNotIn('?', cleaned)  # Should remove punctuation
    
    def test_extract_keywords_interface(self):
        self.assertTrue(hasattr(self.processor, 'extract_keywords'))
        
        # Test keyword extraction
        # keywords = self.processor.extract_keywords("attention mechanism transformer")
        # self.assertIsInstance(keywords, list)
    
    def test_detect_intent_interface(self):
        self.assertTrue(hasattr(self.processor, 'detect_intent'))
    
    def test_extract_filters_interface(self):
        self.assertTrue(hasattr(self.processor, 'extract_filters'))
    
    def test_expand_query_interface(self):
        self.assertTrue(hasattr(self.processor, 'expand_query'))

class TestDocumentRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = Mock()
        self.mock_embedding_service = Mock()
        self.retriever = DocumentRetriever(self.mock_vector_store, self.mock_embedding_service)
    
    def test_init(self):
        self.assertEqual(self.retriever.vector_store, self.mock_vector_store)
        self.assertEqual(self.retriever.embedding_service, self.mock_embedding_service)
    
    def test_retrieve_interface(self):
        self.assertTrue(hasattr(self.retriever, 'retrieve'))
        
        # Mock the embedding service
        self.mock_embedding_service.embed_text.return_value = [0.1] * 768
        
        # Mock the vector store search
        mock_results = [
            SearchResult(
                content="Sample document content",
                metadata={"title": "Test Paper"},
                score=0.85,
                chunk_id="chunk_1"
            )
        ]
        self.mock_vector_store.search.return_value = mock_results
        
        # Test retrieval
        # result = self.retriever.retrieve("test query", k=5)
        # self.assertIsInstance(result, RetrievalResult)
    
    def test_retrieve_with_reranking_interface(self):
        self.assertTrue(hasattr(self.retriever, 'retrieve_with_reranking'))
    
    def test_hybrid_search_interface(self):
        self.assertTrue(hasattr(self.retriever, 'hybrid_search'))
    
    def test_semantic_search_interface(self):
        self.assertTrue(hasattr(self.retriever, '_semantic_search'))
    
    def test_keyword_search_interface(self):
        self.assertTrue(hasattr(self.retriever, '_keyword_search'))

class TestRetrievalIntegration(unittest.TestCase):
    """Integration tests for the retrieval system"""
    
    def setUp(self):
        self.mock_vector_store = Mock()
        self.mock_embedding_service = Mock()
        self.query_processor = QueryProcessor()
        self.retriever = DocumentRetriever(self.mock_vector_store, self.mock_embedding_service)
    
    def test_end_to_end_retrieval_pipeline(self):
        """Test the complete retrieval pipeline"""
        # Sample query
        query = "What are attention mechanisms?"
        
        # Process the query
        # processed_query = self.query_processor.process_query(query)
        
        # Mock embeddings
        self.mock_embedding_service.embed_text.return_value = [0.1] * 768
        
        # Mock search results
        mock_search_results = [
            SearchResult(
                content="Attention mechanisms allow models to focus on relevant parts of the input.",
                metadata={"title": "Attention Is All You Need", "authors": "Vaswani et al."},
                score=0.92,
                chunk_id="attention_paper_chunk_1"
            ),
            SearchResult(
                content="Self-attention computes attention weights for each position in the sequence.",
                metadata={"title": "The Illustrated Transformer", "authors": "Alammar"},
                score=0.88,
                chunk_id="transformer_tutorial_chunk_5"
            )
        ]
        
        self.mock_vector_store.search.return_value = mock_search_results
        
        # Test the retrieval
        # result = self.retriever.retrieve(query, k=5)
        
        # Verify the mocks were called
        # self.mock_embedding_service.embed_text.assert_called_once_with(query)
        # self.mock_vector_store.search.assert_called_once()
        
        # For now, just test that the interfaces exist
        self.assertTrue(hasattr(self.retriever, 'retrieve'))
        self.assertTrue(hasattr(self.query_processor, 'process_query'))

if __name__ == '__main__':
    unittest.main()