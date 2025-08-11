# AI Papers RAG System - Setup Guide

This guide will help you set up the AI Papers RAG system from scratch.

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Git
- Docker (optional, for containerized deployment)
- At least 8GB RAM (16GB recommended)
- 10GB of free disk space

### API Keys Required

- **OpenAI API Key** (required for embeddings and LLM)
- **Anthropic API Key** (optional, for Claude models)
- **Cohere API Key** (optional, for Cohere embeddings)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai_papers_rag
```

### 2. Set Up Python Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Using conda (Alternative)

```bash
# Create conda environment
conda create -n rag-system python=3.10
conda activate rag-system
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For development (includes testing and linting tools)
pip install -r requirements-dev.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys and configuration
nano .env  # or use your preferred editor
```

Required environment variables:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Vector Database
VECTOR_DB_PATH=./data/vector_db
VECTOR_DB_TYPE=chromadb

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=0.7

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

# Optional: Additional API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
```

### 5. Initialize the System

```bash
# Create necessary directories
mkdir -p data/raw_papers data/processed data/vector_db

# Initialize the vector database
python scripts/setup_database.py

# Test the installation
python scripts/test_pipeline.py
```

### 6. Verify Installation

```bash
# Run basic tests
python -m pytest tests/ -v

# Check system status
python scripts/test_pipeline.py --component all
```

## Configuration

### Model Configuration

Edit `configs/model_config.yaml` to customize:

- **LLM models**: Choose between GPT-3.5, GPT-4, Claude, or local models
- **Embedding models**: OpenAI, Sentence-BERT, or Cohere
- **Performance settings**: Caching, rate limits, batch processing
- **Quality settings**: Content filtering, validation, fallback behavior

### Chunking Configuration

Edit `configs/chunking_config.yaml` to customize:

- **Chunking strategy**: Fixed tokens/words, sentences, paragraphs, or sections
- **Chunk size and overlap**: Optimize for your document types
- **Text processing**: Cleaning, preprocessing, language handling
- **Quality control**: Validation, filtering, metadata extraction

### Retrieval Configuration

Edit `configs/retrieval_config.yaml` to customize:

- **Search strategy**: Semantic, keyword, hybrid, or ensemble
- **Retrieval parameters**: Number of results, similarity threshold
- **Re-ranking**: Cross-encoder models for improved relevance
- **Performance**: Caching, indexing, search optimization

## First-Time Setup

### 1. Add Documents

```bash
# Place your PDF files in the raw papers directory
cp your_papers/*.pdf data/raw_papers/

# Or create subdirectories for organization
mkdir -p data/raw_papers/transformers
mkdir -p data/raw_papers/nlp
cp transformers_papers/*.pdf data/raw_papers/transformers/
```

### 2. Ingest Documents

```bash
# Ingest all documents from raw_papers directory
python scripts/ingest_papers.py

# Ingest from specific directory
python scripts/ingest_papers.py --input-dir data/raw_papers/transformers

# Ingest with custom settings
python scripts/ingest_papers.py --batch-size 50 --recursive
```

### 3. Start the Application

```bash
# Start the Streamlit web interface
streamlit run app/main.py

# The application will be available at http://localhost:8501
```

## Docker Setup (Alternative)

### 1. Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Using Docker Only

```bash
# Build the image
docker build -t ai-papers-rag .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ai-papers-rag
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Make sure you're in the correct virtual environment
which python
pip list | grep streamlit

# Reinstall dependencies if needed
pip install --force-reinstall -r requirements.txt
```

#### 2. API Key Issues

```bash
# Check if environment variables are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

# Test API connectivity
python scripts/test_pipeline.py --component llm
```

#### 3. Vector Database Issues

```bash
# Reset the vector database
rm -rf data/vector_db/
python scripts/setup_database.py

# Check database status
python scripts/setup_database.py --test
```

#### 4. Memory Issues

```bash
# Reduce batch size in configuration
export BATCH_SIZE=16

# Use CPU-only mode for embeddings
export DEVICE=cpu

# Monitor memory usage
pip install psutil
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### Performance Issues

#### 1. Slow Response Times

- Enable caching in `configs/model_config.yaml`
- Reduce `RETRIEVAL_K` in `.env`
- Use local embedding models for development
- Consider using FAISS instead of ChromaDB for large datasets

#### 2. Poor Retrieval Quality

- Adjust `SIMILARITY_THRESHOLD` in `.env`
- Experiment with different chunking strategies
- Enable re-ranking in `configs/retrieval_config.yaml`
- Try hybrid search combining semantic and keyword search

#### 3. High API Costs

- Enable embedding caching
- Use local models for development
- Implement batch processing
- Monitor token usage with `log_token_usage: true`

## Development Setup

### 1. Additional Development Tools

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Install Jupyter for notebooks
pip install jupyter
```

### 2. Code Quality

```bash
# Format code
black src/ app/ scripts/
isort src/ app/ scripts/

# Lint code
flake8 src/ app/ scripts/
pylint src/ app/ scripts/

# Type checking
mypy src/ app/ scripts/
```

### 3. Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_document_processor.py
python -m pytest tests/test_retrieval.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Production Deployment

### 1. Environment Setup

```bash
# Use production environment
export ENVIRONMENT=production

# Set up monitoring
pip install prometheus-client
export ENABLE_METRICS=true

# Configure logging
export LOG_LEVEL=INFO
export LOG_FILE=./logs/rag-system.log
```

### 2. Security Considerations

- Use environment variables for all API keys
- Enable HTTPS for web interface
- Implement rate limiting
- Set up API authentication
- Regular security updates

### 3. Monitoring

```bash
# Enable application metrics
export ENABLE_MONITORING=true

# Set up health checks
export HEALTH_CHECK_ENDPOINT=/health

# Configure alerting
export ALERT_WEBHOOK_URL=your_webhook_url
```

## Next Steps

1. **Explore the Notebooks**: Check out the Jupyter notebooks in `/notebooks` for data exploration and experimentation
2. **Customize Configuration**: Adjust settings in `/configs` based on your specific needs
3. **Add More Documents**: Continuously add new papers to improve the knowledge base
4. **Monitor Performance**: Use the evaluation scripts to track system performance
5. **Contribute**: Consider contributing improvements back to the project

## Support

For issues and questions:

1. Check the [API documentation](api_documentation.md)
2. Review the [evaluation results](evaluation_results.md)
3. Search existing GitHub issues
4. Create a new issue with detailed information

## Useful Commands Reference

```bash
# System management
python scripts/setup_database.py          # Initialize database
python scripts/ingest_papers.py          # Add documents
python scripts/test_pipeline.py          # Test system
python scripts/evaluate_system.py        # Run evaluation

# Application
streamlit run app/main.py                # Start web interface
docker-compose up                        # Start with Docker

# Development
python -m pytest tests/                  # Run tests
black src/ && isort src/                 # Format code
jupyter lab                              # Start Jupyter

# Monitoring
tail -f logs/rag-system.log             # View logs
htop                                     # Monitor resources
```