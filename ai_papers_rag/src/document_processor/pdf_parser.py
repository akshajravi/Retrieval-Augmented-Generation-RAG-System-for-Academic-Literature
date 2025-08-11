from typing import List, Dict, Any
from pathlib import Path
import pypdf
import re
from dataclasses import dataclass

@dataclass
class DocumentMetadata:
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    publication_date: str
    source_file: str

class PDFParser:
    def __init__(self):
        self.supported_formats = [".pdf"]
    
    def extract_text(self, file_path: Path) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num,page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
                print(f"Error extracting text from {file_path}: {str(e)}")
                return ""
        return text
    
    def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)

                pdf_metadata = reader.metadata or {}

                first_pages_text = ""

                for i in range(min(3,len(reader.pages))):
                    first_pages_text += reader.pages[i].extract_text()
                
                if pdf_metadata:
                    title = getattr(pdf_metadata, 'title', None) or self._extract_title_from_text(first_pages_text)
                    author = getattr(pdf_metadata, 'author', None) or ""
                    creation_date = str(getattr(pdf_metadata, 'creation_date', None)) if getattr(pdf_metadata, 'creation_date', None) else ""
                else:
                    title = self._extract_title_from_text(first_pages_text)
                    author = ""
                    creation_date = ""
                
                authors = self._extract_authors(author, first_pages_text)
                
                abstract = self._extract_abstract(first_pages_text)
                keywords = self._extract_keywords(first_pages_text)

                return DocumentMetadata(
                    title=title or file_path.stem, 
                    authors=authors,
                    abstract=abstract,
                    keywords=keywords,
                    publication_date=creation_date,
                    source_file=str(file_path)
                )
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {str(e)}")
            # Return basic metadata with filename
            return DocumentMetadata(
                title=file_path.stem,
                authors=[],
                abstract="",
                keywords=[],
                publication_date="",
                source_file=str(file_path)
            )
    
    def parse_document(self, file_path: Path) -> Dict[str, Any]:
        """Parse document and return both text content and metadata."""
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        text_content = self.extract_text(file_path)
        metadata = self.extract_metadata(file_path)
        
        return {
            "content": text_content,
            "metadata": metadata,
            "file_path": str(file_path),
            "file_name": file_path.name
        }
    
    def _extract_title_from_text(self, text: str) -> str:
        """Extract title from first few lines of text."""
        lines = text.split('\n')[:10]  # First 10 lines
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # Reasonable title length
                return line
        return ""
   
    def _extract_authors(self, pdf_author: str, text: str) -> List[str]:
        """Extract authors from PDF metadata or text."""
        # Try PDF metadata first
        if pdf_author:
            return [author.strip() for author in pdf_author.split(',')]
        
        # Basic pattern matching for authors in text
        author_patterns = [
            r'Authors?:\s*([^\n]+)',
            r'By\s+([^\n]+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                authors_text = match.group(1)
                return [author.strip() for author in authors_text.split(',')]
        
        return []
   
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from text."""
        abstract_patterns = [
            r'Abstract[:\-\s]*([^\.]+(?:\.[^\.]*){1,10})',
            r'ABSTRACT[:\-\s]*([^\.]+(?:\.[^\.]*){1,10})',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Limit length to reasonable abstract size
                if len(abstract) < 2000:
                    return abstract
        
        return ""
   
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        keyword_patterns = [
            r'Keywords?[:\-\s]*([^\n]+)',
            r'Key words?[:\-\s]*([^\n]+)',
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords_text = match.group(1)
                keywords = [kw.strip() for kw in re.split(r'[,;]', keywords_text)]
                return [kw for kw in keywords if kw]  # Remove empty strings
        
        return []