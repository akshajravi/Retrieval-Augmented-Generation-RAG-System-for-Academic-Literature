# AI Papers RAG System - API Documentation

This document provides detailed API documentation for the AI Papers RAG system.

## Overview

The RAG system provides both a web interface (Streamlit) and programmatic APIs for document ingestion, retrieval, and question answering.

## Core Components API

### Document Processor

#### PDFParser

```python
from src.document_processor.pdf_parser import PDFParser, DocumentMetadata

parser = PDFParser()

# Extract text from PDF
text = parser.extract_text(file_path="path/to/document.pdf")

# Extract metadata
metadata = parser.extract_metadata(file_path="path/to/document.pdf")

# Parse complete document
result = parser.parse_document(file_path="path/to/document.pdf")
```

**Methods:**

- `extract_text(file_path: Path) -> str`: Extract plain text from PDF
- `extract_metadata(file_path: Path) -> DocumentMetadata`: Extract document metadata
- `parse_document(file_path: Path) -> Dict[str, Any]`: Complete document parsing

**DocumentMetadata Fields:**
```python
@dataclass
class DocumentMetadata:
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    publication_date: str
    source_file: str
```

#### TextChunker

```python
from src.document_processor.chunker import TextChunker, TextChunk

chunker = TextChunker(chunk_size=1000, chunk_overlap=200)

# Chunk text
chunks = chunker.chunk_text(text, metadata={"source": "document.pdf"})

# Advanced chunking methods
token_chunks = chunker.chunk_by_tokens(text, metadata)
sentence_chunks = chunker.chunk_by_sentences(text, metadata)
```

**Methods:**

- `chunk_text(text: str, metadata: Dict = None) -> List[TextChunk]`: Basic text chunking
- `chunk_by_tokens(text: str, metadata: Dict = None) -> List[TextChunk]`: Token-based chunking
- `chunk_by_sentences(text: str, metadata: Dict = None) -> List[TextChunk]`: Sentence-based chunking

**TextChunk Structure:**
```python
@dataclass
class TextChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_index: int
    end_index: int
```

### Embedding Service

#### OpenAI Embedding Service

```python
from src.embeddings.embedding_service import OpenAIEmbeddingService

service = OpenAIEmbeddingService(
    model="text-embedding-ada-002",
    api_key="your-api-key"
)

# Generate single embedding
embedding = service.embed_text("Your text here")

# Generate batch embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = service.embed_batch(texts)
```

#### HuggingFace Embedding Service

```python
from src.embeddings.embedding_service import HuggingFaceEmbeddingService

service = HuggingFaceEmbeddingService(model_name="all-MiniLM-L6-v2")

embedding = service.embed_text("Your text here")
embeddings = service.embed_batch(texts)
```

**Common Methods:**
- `embed_text(text: str) -> List[float]`: Generate single text embedding
- `embed_batch(texts: List[str]) -> List[List[float]]`: Generate batch embeddings

### Vector Store

#### ChromaVectorStore

```python
from src.embeddings.vector_store import ChromaVectorStore, SearchResult

store = ChromaVectorStore(
    collection_name="papers",
    persist_directory="./data/vector_db"
)

# Add documents
documents = ["Document 1 text", "Document 2 text"]
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
metadata = [{"title": "Paper 1"}, {"title": "Paper 2"}]
ids = ["doc1", "doc2"]

store.add_documents(documents, embeddings, metadata, ids)

# Search
query_embedding = [0.15, 0.25, ...]
results = store.search(query_embedding, k=5)
```

#### FAISS Vector Store

```python
from src.embeddings.vector_store import FAISSVectorStore

store = FAISSVectorStore(dimension=1536)
# Same interface as ChromaVectorStore
```

**Methods:**
- `add_documents(documents, embeddings, metadata, ids)`: Add documents to store
- `search(query_embedding, k=5) -> List[SearchResult]`: Search for similar documents
- `delete_documents(ids)`: Remove documents from store

**SearchResult Structure:**
```python
@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
```

### Retrieval System

#### DocumentRetriever

```python
from src.retrieval.retriever import DocumentRetriever
from src.retrieval.query_processor import QueryProcessor

query_processor = QueryProcessor()
retriever = DocumentRetriever(vector_store, embedding_service)

# Basic retrieval
results = retriever.retrieve("What is attention mechanism?", k=5)

# Advanced retrieval with filtering
results = retriever.retrieve(
    query="transformer architecture",
    k=10,
    filter_metadata={"document_type": "research_paper"}
)

# Hybrid search
results = retriever.hybrid_search("BERT vs GPT", k=5, alpha=0.7)
```

**Methods:**
- `retrieve(query, k=5, filter_metadata=None) -> RetrievalResult`: Basic retrieval
- `retrieve_with_reranking(query, k=5, rerank_top_k=20) -> RetrievalResult`: With re-ranking
- `hybrid_search(query, k=5, alpha=0.5) -> RetrievalResult`: Semantic + keyword search

### LLM Integration

#### OpenAI LLM Client

```python
from src.llm.llm_client import OpenAILLMClient, LLMResponse

client = OpenAILLMClient(
    model="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Generate response
response = client.generate_response(
    prompt="Explain transformers",
    temperature=0.1,
    max_tokens=500
)

# Generate with context
context = ["Context document 1", "Context document 2"]
response = client.generate_with_context(
    query="What is attention?",
    context=context,
    temperature=0.1
)
```

**LLMResponse Structure:**
```python
@dataclass
class LLMResponse:
    content: str
    model_used: str
    tokens_used: int
    finish_reason: str
```

### RAG Pipeline

#### Complete Pipeline

```python
from src.rag_pipeline.pipeline import RAGPipeline
from src.rag_pipeline.response_formatter import ResponseFormatter

# Initialize pipeline
pipeline = RAGPipeline(
    retriever=retriever,
    llm_client=llm_client,
    query_processor=query_processor
)

# Query the system
response = pipeline.query(
    user_query="How does BERT work?",
    k=5,
    include_sources=True
)

# Format response
formatter = ResponseFormatter()
formatted = formatter.format_response(
    response.answer,
    response.retrieval_results,
    include_citations=True
)
```

**RAGResponse Structure:**
```python
@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    processing_time: float
    retrieval_results: RetrievalResult
    llm_response: LLMResponse
```

## Configuration API

### Loading Configuration

```python
from src.config import Config
import yaml

# Access environment-based config
print(Config.CHUNK_SIZE)
print(Config.LLM_MODEL)

# Load YAML configurations
with open('configs/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

with open('configs/retrieval_config.yaml', 'r') as f:
    retrieval_config = yaml.safe_load(f)
```

### Dynamic Configuration Updates

```python
# Update configuration at runtime
Config.RETRIEVAL_K = 10
Config.SIMILARITY_THRESHOLD = 0.8

# Reload configuration
import importlib
import src.config
importlib.reload(src.config)
```

## Script APIs

### Database Setup

```python
# Programmatic database setup
from scripts.setup_database import setup_chromadb, setup_embedding_service

# Set up vector store
vector_store = setup_chromadb()

# Set up embedding service
embedding_service = setup_embedding_service()
```

### Document Ingestion

```python
from scripts.ingest_papers import DocumentIngester

ingester = DocumentIngester()

# Ingest single file
result = ingester.ingest_file(Path("paper.pdf"))

# Ingest directory
results = ingester.ingest_directory(
    input_dir=Path("data/raw_papers"),
    recursive=True
)

# Get statistics
stats = ingester.get_ingestion_stats()
```

### System Testing

```python
from scripts.test_pipeline import PipelineTester

tester = PipelineTester()

# Test individual components
embedding_ok = tester.test_embedding_service()
vector_ok = tester.test_vector_store()
llm_ok = tester.test_llm_client()
pipeline_ok = tester.test_rag_pipeline()

# Run benchmarks
tester.test_performance_benchmarks()
```

### System Evaluation

```python
from scripts.evaluate_system import RAGEvaluator

evaluator = RAGEvaluator()

# Run evaluation
results = evaluator.evaluate_system_performance()

# Generate report
evaluator.generate_evaluation_report(
    output_file=Path("evaluation_results.json")
)

# Compare with baseline
evaluator.compare_with_baseline(Path("baseline_results.json"))
```

## Error Handling

### Common Exceptions

```python
from src.exceptions import (
    EmbeddingServiceError,
    VectorStoreError,
    LLMClientError,
    RAGPipelineError,
    DocumentProcessingError
)

try:
    response = pipeline.query("Your question")
except EmbeddingServiceError as e:
    print(f"Embedding error: {e}")
except VectorStoreError as e:
    print(f"Vector store error: {e}")
except LLMClientError as e:
    print(f"LLM error: {e}")
except RAGPipelineError as e:
    print(f"Pipeline error: {e}")
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_query(pipeline, query):
    return pipeline.query(query)
```

## Performance Optimization

### Caching

```python
from functools import lru_cache
import pickle

# Cache embeddings
@lru_cache(maxsize=1000)
def cached_embed_text(text):
    return embedding_service.embed_text(text)

# Persistent caching
def save_embeddings_cache(cache, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(cache, f)

def load_embeddings_cache(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
```

### Batch Processing

```python
# Batch document processing
def batch_ingest_documents(file_paths, batch_size=32):
    results = []
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        batch_results = []
        
        for file_path in batch:
            result = ingester.ingest_file(file_path)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results
```

### Async Operations

```python
import asyncio
import aiohttp

async def async_embed_batch(texts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            task = async_embed_single(session, text)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

# Usage
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = await async_embed_batch(texts)
```

## Monitoring and Logging

### Logging Configuration

```python
import logging
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger(
    name="rag_system",
    level=logging.INFO,
    log_file="rag_system.log"
)

# Use in your code
logger.info("Starting RAG query processing")
logger.warning("Low similarity scores detected")
logger.error("Failed to process document", exc_info=True)
```

### Metrics Collection

```python
from src.utils.metrics import MetricsCollector

metrics = MetricsCollector()

# Collect performance metrics
with metrics.timer("query_processing"):
    response = pipeline.query(query)

# Collect custom metrics
metrics.increment("queries_processed")
metrics.histogram("similarity_scores", scores)
metrics.gauge("active_connections", connection_count)

# Export metrics
prometheus_metrics = metrics.export_prometheus()
json_metrics = metrics.export_json()
```

## Integration Examples

### Flask API Integration

```python
from flask import Flask, request, jsonify
from src.rag_pipeline.pipeline import RAGPipeline

app = Flask(__name__)
pipeline = RAGPipeline(retriever, llm_client)

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.json
    query = data.get('query')
    k = data.get('k', 5)
    
    try:
        response = pipeline.query(query, k=k)
        return jsonify({
            'answer': response.answer,
            'sources': response.sources,
            'processing_time': response.processing_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})
```

### Streamlit Integration

```python
import streamlit as st
from src.rag_pipeline.pipeline import RAGPipeline

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return RAGPipeline(retriever, llm_client)

pipeline = get_pipeline()

# Streamlit interface
query = st.text_input("Enter your question:")

if st.button("Search"):
    with st.spinner("Searching..."):
        response = pipeline.query(query)
        
        st.write("**Answer:**")
        st.write(response.answer)
        
        st.write("**Sources:**")
        for i, source in enumerate(response.sources, 1):
            st.write(f"{i}. {source['title']} (Score: {source['score']:.3f})")
```

## Best Practices

### Error Handling

```python
def robust_pipeline_query(pipeline, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return pipeline.query(query)
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
```

### Resource Management

```python
from contextlib import contextmanager

@contextmanager
def rag_pipeline_context():
    pipeline = RAGPipeline(retriever, llm_client)
    try:
        yield pipeline
    finally:
        # Clean up resources
        pipeline.cleanup()

# Usage
with rag_pipeline_context() as pipeline:
    response = pipeline.query("Your question")
```

### Performance Monitoring

```python
import time
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    query_time: float
    retrieval_time: float
    generation_time: float
    total_tokens: int
    cost_estimate: float

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        metrics = PerformanceMetrics(
            query_time=end_time - start_time,
            retrieval_time=getattr(result, 'retrieval_time', 0),
            generation_time=getattr(result, 'generation_time', 0),
            total_tokens=getattr(result, 'tokens_used', 0),
            cost_estimate=calculate_cost(result)
        )
        
        log_metrics(metrics)
        return result
    
    return wrapper
```

This API documentation provides comprehensive coverage of all system components and their programmatic interfaces. For additional examples and advanced usage patterns, refer to the example scripts and Jupyter notebooks in the repository.