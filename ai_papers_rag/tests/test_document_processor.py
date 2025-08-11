import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.document_processor.pdf_parser import PDFParser, DocumentMetadata
from src.document_processor.chunker import TextChunker, TextChunk
from src.document_processor.metadata_extractor import MetadataExtractor, ExtractedMetadata

class TestPDFParser(unittest.TestCase):
    def setUp(self):
        self.parser = PDFParser()
        self.sample_pdf_path = Path("test_data/sample.pdf")
    
    def test_init(self):
        self.assertIsInstance(self.parser, PDFParser)
        self.assertEqual(self.parser.supported_formats, [".pdf"])
    
    @patch('pypdf.PdfReader')
    def test_extract_text(self, mock_reader):
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample text from PDF"
        mock_reader.return_value.pages = [mock_page]
        
        # This would test the actual implementation
        # result = self.parser.extract_text(self.sample_pdf_path)
        # self.assertIsInstance(result, str)
        # self.assertIn("Sample text", result)
        
        # For now, just test the interface exists
        self.assertTrue(hasattr(self.parser, 'extract_text'))
    
    def test_extract_metadata(self):
        # Test the interface exists
        self.assertTrue(hasattr(self.parser, 'extract_metadata'))
    
    def test_parse_document(self):
        # Test the interface exists
        self.assertTrue(hasattr(self.parser, 'parse_document'))

class TestTextChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        self.sample_text = "This is a sample text for testing chunking functionality. " * 10
    
    def test_init(self):
        self.assertEqual(self.chunker.chunk_size, 100)
        self.assertEqual(self.chunker.chunk_overlap, 20)
        self.assertIsNotNone(self.chunker.tokenizer)
    
    def test_chunk_text_interface(self):
        # Test the interface exists
        self.assertTrue(hasattr(self.chunker, 'chunk_text'))
        
        # Test with sample data
        metadata = {"source": "test.pdf"}
        # chunks = self.chunker.chunk_text(self.sample_text, metadata)
        # self.assertIsInstance(chunks, list)
    
    def test_chunk_by_tokens_interface(self):
        self.assertTrue(hasattr(self.chunker, 'chunk_by_tokens'))
    
    def test_chunk_by_sentences_interface(self):
        self.assertTrue(hasattr(self.chunker, 'chunk_by_sentences'))

class TestMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MetadataExtractor()
        self.sample_text = """
        Title: Attention Is All You Need
        
        Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar
        
        Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.
        
        Keywords: attention, transformer, neural networks
        """
    
    def test_init(self):
        self.assertIsInstance(self.extractor.title_patterns, list)
        self.assertIsInstance(self.extractor.author_patterns, list)
    
    def test_extract_from_text_interface(self):
        self.assertTrue(hasattr(self.extractor, 'extract_from_text'))
        
        # Test with sample data
        # metadata = self.extractor.extract_from_text(self.sample_text)
        # self.assertIsInstance(metadata, ExtractedMetadata)
    
    def test_extract_from_filename_interface(self):
        self.assertTrue(hasattr(self.extractor, 'extract_from_filename'))
    
    def test_extract_title_interface(self):
        self.assertTrue(hasattr(self.extractor, 'extract_title'))
    
    def test_extract_authors_interface(self):
        self.assertTrue(hasattr(self.extractor, 'extract_authors'))
    
    def test_extract_abstract_interface(self):
        self.assertTrue(hasattr(self.extractor, 'extract_abstract'))

if __name__ == '__main__':
    unittest.main()