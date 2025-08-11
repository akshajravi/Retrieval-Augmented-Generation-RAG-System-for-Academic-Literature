# AI Papers RAG System (Still in Progress)

A Retrieval-Augmented Generation (RAG) system for querying and analyzing AI research papers. This system allows you to upload research papers, index them in a vector database, and query them using natural language.

## Features

- PDF document processing and text extraction
- Intelligent text chunking and metadata extraction
- Vector embeddings with ChromaDB/FAISS support
- Semantic search and retrieval
- LLM-powered answer generation with source citations
- Interactive Streamlit web interface
- Docker support for easy deployment

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai_papers_rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and configuration
```

5. Initialize the vector database:
```bash
python scripts/setup_database.py
```

### Usage

#### Running the Streamlit App

```bash
streamlit run app/main.py
```

#### Ingesting Documents

```bash
python scripts/ingest_papers.py --input-dir data/raw_papers
```

#### Testing the Pipeline

```bash
python scripts/test_pipeline.py
```

### Docker Deployment

```bash
docker-compose up --build
```

## Project Structure

```
ai_papers_rag/
├── src/                    # Core library code
│   ├── document_processor/ # PDF parsing and chunking
│   ├── embeddings/        # Vector embeddings and storage
│   ├── retrieval/         # Search and retrieval logic
│   ├── llm/              # Language model integration
│   └── rag_pipeline/     # End-to-end RAG pipeline
├── app/                   # Streamlit application
├── scripts/              # Utility scripts
├── tests/                # Unit and integration tests
├── notebooks/            # Jupyter notebooks for experimentation
├── configs/              # YAML configuration files
└── docs/                 # Documentation
```

## Configuration

The system uses environment variables and YAML files for configuration:

- `.env`: Environment variables (API keys, paths)
- `configs/model_config.yaml`: LLM and embedding model settings
- `configs/chunking_config.yaml`: Text chunking parameters
- `configs/retrieval_config.yaml`: Search and retrieval settings

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

```bash
black src/ app/ scripts/
flake8 src/ app/ scripts/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
