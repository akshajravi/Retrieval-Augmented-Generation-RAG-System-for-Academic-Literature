#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, TaskID

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.document_processor.pdf_parser import PDFParser
from src.document_processor.chunker import TextChunker
from src.document_processor.metadata_extractor import MetadataExtractor
from src.embeddings.embedding_service import OpenAIEmbeddingService, HuggingFaceEmbeddingService
from src.embeddings.vector_store import VectorStore, FAISSVectorStore

console = Console()

class DocumentIngester:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize embedding service
        if Config.EMBEDDING_MODEL.startswith("text-embedding"):
            self.embedding_service = OpenAIEmbeddingService(
                model=Config.EMBEDDING_MODEL,
                api_key=Config.OPENAI_API_KEY
            )
        else:
            self.embedding_service = HuggingFaceEmbeddingService(
                model_name=Config.EMBEDDING_MODEL
            )
        
        # Initialize vector store
        if Config.VECTOR_DB_TYPE == "chromadb":
            self.vector_store = VectorStore(
                collection_name="ai_papers",
                persist_directory=Config.VECTOR_DB_PATH
            )
        else:
            self.vector_store = FAISSVectorStore(
                dimension=Config.EMBEDDING_DIMENSION
            )
    
    def ingest_directory(self, input_dir: Path, recursive: bool = True) -> Dict[str, Any]:
        console.print(f"[bold blue]Ingesting documents from: {input_dir}[/bold blue]")
        
        # Find all PDF files
        if recursive:
            pdf_files = list(input_dir.rglob("*.pdf"))
        else:
            pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            console.print("[yellow]No PDF files found in the specified directory[/yellow]")
            return {"processed": 0, "errors": 0, "files": []}
        
        console.print(f"[green]Found {len(pdf_files)} PDF files[/green]")
        
        results = {
            "processed": 0,
            "errors": 0,
            "files": [],
            "total_chunks": 0,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        with Progress() as progress:
            task = progress.add_task("Processing documents...", total=len(pdf_files))
            
            for pdf_file in pdf_files:
                try:
                    file_result = self.ingest_file(pdf_file)
                    results["processed"] += 1
                    results["total_chunks"] += file_result.get("chunks", 0)
                    results["files"].append({
                        "file": str(pdf_file),
                        "status": "success",
                        "chunks": file_result.get("chunks", 0)
                    })
                    
                    console.print(f"✓ Processed: {pdf_file.name} ({file_result.get('chunks', 0)} chunks)")
                    
                except Exception as e:
                    results["errors"] += 1
                    results["files"].append({
                        "file": str(pdf_file),
                        "status": "error",
                        "error": str(e)
                    })
                    
                    console.print(f"[red]❌ Failed: {pdf_file.name} - {e}[/red]")
                
                progress.advance(task)
        
        results["processing_time"] = time.time() - start_time
        
        return results
    
    def ingest_file(self, file_path: Path) -> Dict[str, Any]:
        console.print(f"[cyan]Processing: {file_path.name}[/cyan]")
        
        # Extract text and metadata
        # text = self.pdf_parser.extract_text(file_path)
        # metadata = self.pdf_parser.extract_metadata(file_path)
        
        # For now, use placeholder data
        text = f"Sample text content from {file_path.name}"
        metadata = {
            "title": file_path.stem,
            "source_file": str(file_path),
            "file_size": file_path.stat().st_size
        }
        
        # Chunk the text
        # chunks = self.chunker.chunk_text(text, metadata)
        
        # For now, create sample chunks
        chunks = [
            {
                "content": f"Chunk 1 from {file_path.name}: {text[:200]}",
                "metadata": metadata,
                "chunk_id": f"{file_path.stem}_chunk_0"
            },
            {
                "content": f"Chunk 2 from {file_path.name}: {text[100:300]}",
                "metadata": metadata,
                "chunk_id": f"{file_path.stem}_chunk_1"
            }
        ]
        
        # Generate embeddings
        console.print("  Generating embeddings...")
        chunk_texts = [chunk["content"] for chunk in chunks]
        # embeddings = self.embedding_service.embed_batch(chunk_texts)
        
        # For now, use placeholder embeddings
        embeddings = [[0.1] * Config.EMBEDDING_DIMENSION for _ in chunks]
        
        # Store in vector database
        console.print("  Storing in vector database...")
        documents = chunk_texts
        metadata_list = [chunk["metadata"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        
        # self.vector_store.add_documents(documents, embeddings, metadata_list, ids)
        
        return {
            "chunks": len(chunks),
            "text_length": len(text),
            "metadata": metadata
        }
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        # Get final statistics
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "database_size": "Unknown"
        }

def main():
    parser = argparse.ArgumentParser(description="Ingest research papers into the RAG system")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Config.RAW_PAPERS_DIR,
        help="Directory containing PDF files to ingest"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for PDF files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Number of documents to process in each batch"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current ingestion statistics"
    )
    
    args = parser.parse_args()
    
    console.print("[bold green]📚 AI Papers Document Ingestion[/bold green]")
    console.print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Validate API key if using OpenAI
    if Config.EMBEDDING_MODEL.startswith("text-embedding") and not Config.OPENAI_API_KEY:
        console.print("[red]Error: OpenAI API key is required for OpenAI embeddings[/red]")
        console.print("Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    
    # Initialize ingester
    try:
        ingester = DocumentIngester()
    except Exception as e:
        console.print(f"[red]Error initializing ingester: {e}[/red]")
        sys.exit(1)
    
    # Show current stats if requested
    if args.stats:
        stats = ingester.get_ingestion_stats()
        console.print("[bold]Current Database Statistics:[/bold]")
        console.print(f"Total Documents: {stats['total_documents']}")
        console.print(f"Total Chunks: {stats['total_chunks']}")
        console.print(f"Database Size: {stats['database_size']}")
        return
    
    # Check if input directory exists
    if not args.input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {args.input_dir}[/red]")
        sys.exit(1)
    
    # Run ingestion
    try:
        results = ingester.ingest_directory(args.input_dir, args.recursive)
        
        # Print results
        console.print("\n[bold]Ingestion Results:[/bold]")
        console.print(f"Processed: {results['processed']} files")
        console.print(f"Errors: {results['errors']} files")
        console.print(f"Total Chunks: {results['total_chunks']}")
        console.print(f"Processing Time: {results['processing_time']:.2f} seconds")
        
        if results['errors'] > 0:
            console.print("\n[yellow]Files with errors:[/yellow]")
            for file_info in results['files']:
                if file_info['status'] == 'error':
                    console.print(f"  {file_info['file']}: {file_info['error']}")
        
        console.print("\n[green]✅ Ingestion completed![/green]")
        
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()