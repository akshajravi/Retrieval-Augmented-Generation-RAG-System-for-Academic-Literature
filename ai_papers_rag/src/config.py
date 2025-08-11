from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Vector Database
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chromadb")
    
    # Document Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_TOKENS_PER_CHUNK: int = int(os.getenv("MAX_TOKENS_PER_CHUNK", "1000"))
    
    # Retrieval
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    RETRIEVAL_STRATEGY: str = os.getenv("RETRIEVAL_STRATEGY", "semantic")
    
    # Language Model
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Application
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # File Processing
    ALLOWED_EXTENSIONS: list = os.getenv("ALLOWED_EXTENSIONS", "pdf,txt,md").split(",")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # Performance
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
    PROCESSED_DIR = DATA_DIR / "processed"
    CONFIGS_DIR = PROJECT_ROOT / "configs"