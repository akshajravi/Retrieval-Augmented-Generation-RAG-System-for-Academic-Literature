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

# Load environment variables
load_dotenv()

class BasicIngester:
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
        
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return ""
        return text
    
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
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                print(f"  Could not extract text from {pdf_file.name}")
                continue
            
            # Create simple chunks
            chunks = self.simple_chunk_text(text, chunk_size=1000)
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
                metadatas.append({
                    "source_file": pdf_file.name,
                    "title": pdf_file.stem,
                    "chunk_id": i,
                    "authors": "Unknown"
                })
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
            print(f"\n🎉 Ingestion complete!")
            print(f"Total documents in database: {count}")
        except Exception as e:
            print(f"\nError getting stats: {e}")

def main():
    print("🚀 Basic Document Ingester")
    print("=" * 40)
    
    try:
        ingester = BasicIngester()
        
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