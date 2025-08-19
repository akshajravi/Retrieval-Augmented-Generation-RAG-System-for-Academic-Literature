#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load environment variables
load_dotenv()

from src.config import Config
from src.document_processor.pdf_parser import PDFParser
from src.document_processor.chunker import TextChunker
# Import vector store directly to avoid embedding service issues
from src.embeddings.vector_store import VectorStore, SearchResult
import openai

class SimpleDocumentIngester:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name="ai_papers",
            persist_directory=Config.VECTOR_DB_PATH
        )
        
        # Set up OpenAI client
        openai.api_key = Config.OPENAI_API_KEY
        
    def get_openai_embedding(self, text: str):
        """Get embedding using OpenAI API directly"""
        try:
            response = openai.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def ingest_papers(self, papers_dir: Path):
        """Ingest all papers from the directory"""
        pdf_files = list(papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {papers_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            
            # Extract text
            text = self.pdf_parser.extract_text(pdf_file)
            if not text:
                print(f"  Could not extract text from {pdf_file.name}")
                continue
            
            # Create chunks
            metadata = {
                "source_file": pdf_file.name,
                "title": pdf_file.stem,
                "authors": ["Unknown"],
            }
            
            chunks = self.chunker.chunk_text(text, metadata)
            print(f"  Created {len(chunks)} chunks")
            
            # Process chunks in batches
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)}", end="\r")
                
                # Get embedding
                embedding = self.get_openai_embedding(chunk.content)
                if embedding is None:
                    continue
                
                documents.append(chunk.content)
                embeddings.append(embedding)
                metadatas.append(chunk.metadata)
                ids.append(f"{pdf_file.stem}_chunk_{i}")
                
                # Add to vector store in batches of 10
                if len(documents) >= 10:
                    self.vector_store.add_documents(documents, embeddings, metadatas, ids)
                    documents, embeddings, metadatas, ids = [], [], [], []
                
                time.sleep(0.1)  # Rate limiting
            
            # Add remaining documents
            if documents:
                self.vector_store.add_documents(documents, embeddings, metadatas, ids)
            
            print(f"\n  ✓ Finished processing {pdf_file.name}")
        
        # Get final stats
        stats = self.vector_store.get_stats()
        print(f"\n🎉 Ingestion complete!")
        print(f"Total documents in database: {stats.get('total_documents', 0)}")

def main():
    print("🚀 Simple Document Ingester")
    print("=" * 40)
    
    # Check if OpenAI API key is set
    if not Config.OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file")
        return
    
    ingester = SimpleDocumentIngester()
    
    # Check papers directory
    papers_dir = Config.RAW_PAPERS_DIR
    if not papers_dir.exists():
        print(f"❌ Error: Papers directory not found: {papers_dir}")
        return
    
    # Start ingestion
    ingester.ingest_papers(papers_dir)

if __name__ == "__main__":
    main()