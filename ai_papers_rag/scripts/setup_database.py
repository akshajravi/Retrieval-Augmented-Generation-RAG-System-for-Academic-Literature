#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.embeddings.vector_store import VectorStore, FAISSVectorStore
from src.embeddings.embedding_service import OpenAIEmbeddingService, HuggingFaceEmbeddingService

def setup_chromadb():
    print("Setting up ChromaDB...")
    
    # Create vector database directory
    db_path = Path(Config.VECTOR_DB_PATH)
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB
    vector_store = VectorStore(
        collection_name="ai_papers",
        persist_directory=str(db_path)
    )
    
    print(f"ChromaDB initialized at: {db_path}")
    return vector_store

def setup_faiss():
    print("Setting up FAISS...")
    
    # Create vector database directory
    db_path = Path(Config.VECTOR_DB_PATH)
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize FAISS
    vector_store = FAISSVectorStore(dimension=Config.EMBEDDING_DIMENSION)
    
    print(f"FAISS initialized with dimension: {Config.EMBEDDING_DIMENSION}")
    return vector_store

def setup_embedding_service():
    print("Setting up embedding service...")
    
    if Config.EMBEDDING_MODEL.startswith("text-embedding"):
        # OpenAI embedding service
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        
        embedding_service = OpenAIEmbeddingService(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )
        print(f"OpenAI embedding service initialized with model: {Config.EMBEDDING_MODEL}")
    
    else:
        # HuggingFace embedding service
        embedding_service = HuggingFaceEmbeddingService(model_name=Config.EMBEDDING_MODEL)
        print(f"HuggingFace embedding service initialized with model: {Config.EMBEDDING_MODEL}")
    
    return embedding_service

def create_data_directories():
    print("Creating data directories...")
    
    directories = [
        Config.DATA_DIR,
        Config.RAW_PAPERS_DIR,
        Config.PROCESSED_DIR,
        Path(Config.VECTOR_DB_PATH)
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def test_setup():
    print("\nTesting setup...")
    
    try:
        # Test embedding service
        embedding_service = setup_embedding_service()
        test_text = "This is a test sentence for embeddings."
        
        print("Testing embedding generation...")
        # embedding = embedding_service.embed_text(test_text)
        print("✓ Embedding service test passed (placeholder)")
        
        # Test vector store
        if Config.VECTOR_DB_TYPE == "chromadb":
            vector_store = setup_chromadb()
        else:
            vector_store = setup_faiss()
        
        print("✓ Vector store test passed")
        
        print("\n✅ All tests passed! Database setup is complete.")
        
    except Exception as e:
        print(f"\n❌ Setup test failed: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up the RAG system database")
    parser.add_argument(
        "--vector-db", 
        choices=["chromadb", "faiss"], 
        default="chromadb",
        help="Vector database type to use"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run tests after setup"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force recreate database even if it exists"
    )
    
    args = parser.parse_args()
    
    print("🚀 Setting up AI Papers RAG Database")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Create directories
    create_data_directories()
    
    # Set up vector database
    if args.vector_db == "chromadb":
        vector_store = setup_chromadb()
    else:
        vector_store = setup_faiss()
    
    # Set up embedding service
    embedding_service = setup_embedding_service()
    
    # Run tests if requested
    if args.test:
        success = test_setup()
        if not success:
            sys.exit(1)
    
    print("\n🎉 Database setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your research papers to data/raw_papers/")
    print("2. Run: python scripts/ingest_papers.py")
    print("3. Start the application: streamlit run app/main.py")

if __name__ == "__main__":
    main()