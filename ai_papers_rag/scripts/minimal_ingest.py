#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Direct imports to avoid __init__.py issues
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import directly from files
import sys
sys.path.append(str(Path(__file__).parent.parent / "src" / "document_processor"))
sys.path.append(str(Path(__file__).parent.parent / "src" / "embeddings"))

from pdf_parser import PDFParser
from chunker import TextChunker
from vector_store import VectorStore, SearchResult
import openai

class MinimalIngester:
    def __init__(self):
        # Config values directly
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.pdf_parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name="ai_papers",
            persist_directory=self.vector_db_path
        )
        
        # Set up OpenAI
        openai.api_key = self.openai_api_key
        
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
        """Ingest all papers from the directory"""
        papers_path = Path(papers_dir)
        pdf_files = list(papers_path.glob("*.pdf"))
        
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
            
            # Create metadata
            metadata = {
                "source_file": pdf_file.name,
                "title": pdf_file.stem,
                "authors": ["Unknown"],
            }
            
            # Create chunks
            chunks = self.chunker.chunk_text(text, metadata)
            print(f"  Created {len(chunks)} chunks")
            
            # Process chunks
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)}", end="\r")
                
                # Get embedding
                embedding = self.get_embedding(chunk.content)
                if embedding is None:
                    continue
                
                documents.append(chunk.content)
                embeddings.append(embedding)
                metadatas.append(chunk.metadata)
                ids.append(f"{pdf_file.stem}_chunk_{i}")
                
                # Add to vector store in small batches
                if len(documents) >= 5:
                    try:
                        self.vector_store.add_documents(documents, embeddings, metadatas, ids)
                        documents, embeddings, metadatas, ids = [], [], [], []
                    except Exception as e:
                        print(f"\n  Error adding batch: {e}")
                
                time.sleep(0.2)  # Rate limiting
            
            # Add remaining documents
            if documents:
                try:
                    self.vector_store.add_documents(documents, embeddings, metadatas, ids)
                except Exception as e:
                    print(f"\n  Error adding final batch: {e}")
            
            print(f"\n  ✓ Finished processing {pdf_file.name}")
        
        # Get final stats
        try:
            stats = self.vector_store.get_stats()
            print(f"\n🎉 Ingestion complete!")
            print(f"Total documents in database: {stats.get('total_documents', 0)}")
        except Exception as e:
            print(f"\nError getting stats: {e}")

def main():
    print("🚀 Minimal Document Ingester")
    print("=" * 40)
    
    try:
        ingester = MinimalIngester()
        
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