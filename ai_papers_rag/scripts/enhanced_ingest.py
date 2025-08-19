#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv
import openai
import chromadb
import numpy as np
import pypdf
import re

# Load environment variables
load_dotenv()

class EnhancedIngester:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Set up OpenAI
        openai.api_key = self.openai_api_key
        
        # Initialize ChromaDB directly
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        self.collection = self.client.get_or_create_collection(
            name="ai_papers",
            metadata={"description": "Academic papers collection"}
        )
        
    def extract_metadata_from_pdf(self, file_path: Path, first_page_text: str) -> dict:
        """Extract title and authors from PDF metadata and first page"""
        metadata = {
            "title": file_path.stem,  # Default to filename
            "authors": "Unknown"
        }
        
        try:
            # Try to get metadata from PDF
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                
                # Check PDF metadata
                if reader.metadata:
                    if reader.metadata.title:
                        metadata["title"] = reader.metadata.title
                    if reader.metadata.author:
                        metadata["authors"] = reader.metadata.author
        except:
            pass
        
        # If no title from metadata, try to extract from first page
        if metadata["title"] == file_path.stem and first_page_text:
            extracted_title = self.extract_title_from_text(first_page_text)
            if extracted_title:
                metadata["title"] = extracted_title
        
        # Try to extract authors from first page if not found
        if metadata["authors"] == "Unknown" and first_page_text:
            extracted_authors = self.extract_authors_from_text(first_page_text)
            if extracted_authors:
                metadata["authors"] = extracted_authors
        
        return metadata
    
    def extract_title_from_text(self, text: str) -> str:
        """Extract paper title from first page text"""
        lines = text.split('\n')
        
        # Look for title patterns in the first few lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # Skip common headers/footers
            if any(skip in line.lower() for skip in ['arxiv', 'proceedings', 'conference', 'workshop', 'page']):
                continue
            
            # Look for title-like lines (longer than 10 chars, not all caps)
            if len(line) > 10 and not line.isupper() and not line.isdigit():
                # Clean up the title
                title = re.sub(r'^[^\w]*', '', line)  # Remove leading symbols
                title = re.sub(r'[^\w]*$', '', title)  # Remove trailing symbols
                
                if len(title) > 10:
                    return title
        
        return None
    
    def extract_authors_from_text(self, text: str) -> str:
        """Extract authors from first page text"""
        lines = text.split('\n')
        
        # Look for author patterns after title
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            
            # Skip empty lines and common patterns
            if not line or len(line) < 3:
                continue
            
            # Look for patterns that suggest authors
            author_patterns = [
                r'^[A-Z][a-z]+ [A-Z][a-z]+',  # FirstName LastName
                r'[A-Z]\. [A-Z][a-z]+',       # F. LastName
                r'^[A-Z][a-z]+, [A-Z][a-z]+', # LastName, FirstName
            ]
            
            for pattern in author_patterns:
                if re.search(pattern, line):
                    # Clean up the author line
                    authors = re.sub(r'\d+$', '', line)  # Remove trailing numbers
                    authors = re.sub(r'^\d+\s*', '', authors)  # Remove leading numbers
                    authors = authors.strip()
                    
                    if len(authors) > 3 and len(authors) < 100:
                        return authors
        
        return None
    
    def extract_text_from_pdf(self, file_path: Path) -> tuple:
        """Extract text from PDF and return full text + first page"""
        full_text = ""
        first_page_text = ""
        
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_num == 0:
                        first_page_text = page_text
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return "", ""
            
        return full_text, first_page_text
    
    def simple_chunk_text(self, text: str, chunk_size: int = 1000) -> list:
        """Simple text chunking by character count"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_embedding(self, text: str):
        """Get embedding using OpenAI API"""
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def ingest_papers(self, papers_dir: str):
        """Ingest all papers from the directory with enhanced metadata"""
        papers_path = Path(papers_dir)
        pdf_files = list(papers_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {papers_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            
            # Extract text and first page
            full_text, first_page_text = self.extract_text_from_pdf(pdf_file)
            if not full_text:
                print(f"  Could not extract text from {pdf_file.name}")
                continue
            
            # Extract enhanced metadata
            paper_metadata = self.extract_metadata_from_pdf(pdf_file, first_page_text)
            print(f"  Title: {paper_metadata['title']}")
            print(f"  Authors: {paper_metadata['authors']}")
            
            # Create simple chunks
            chunks = self.simple_chunk_text(full_text, chunk_size=1000)
            print(f"  Created {len(chunks)} chunks")
            
            # Process chunks
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)}", end="\r")
                
                # Get embedding
                embedding = self.get_embedding(chunk)
                if embedding is None:
                    continue
                
                documents.append(chunk)
                embeddings.append(embedding)
                
                # Enhanced metadata for each chunk
                chunk_metadata = {
                    "source_file": pdf_file.name,
                    "title": paper_metadata["title"],
                    "authors": paper_metadata["authors"],
                    "chunk_id": i
                }
                metadatas.append(chunk_metadata)
                ids.append(f"{pdf_file.stem}_chunk_{i}")
                
                # Add to ChromaDB in batches of 3
                if len(documents) >= 3:
                    try:
                        self.collection.add(
                            documents=documents,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            ids=ids
                        )
                        documents, embeddings, metadatas, ids = [], [], [], []
                    except Exception as e:
                        print(f"\n  Error adding batch: {e}")
                
                time.sleep(0.3)  # Rate limiting
            
            # Add remaining documents
            if documents:
                try:
                    self.collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e:
                    print(f"\n  Error adding final batch: {e}")
            
            print(f"\n  ✓ Finished processing {pdf_file.name}")
        
        # Get final stats
        try:
            count = self.collection.count()
            print(f"\n🎉 Enhanced ingestion complete!")
            print(f"Total documents in database: {count}")
        except Exception as e:
            print(f"\nError getting stats: {e}")

def main():
    print("🚀 Enhanced Document Ingester with Metadata Extraction")
    print("=" * 60)
    
    try:
        ingester = EnhancedIngester()
        
        # Use default papers directory
        papers_dir = "./data/raw_papers"
        if not Path(papers_dir).exists():
            print(f"❌ Error: Papers directory not found: {papers_dir}")
            return
        
        # Start ingestion
        ingester.ingest_papers(papers_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()