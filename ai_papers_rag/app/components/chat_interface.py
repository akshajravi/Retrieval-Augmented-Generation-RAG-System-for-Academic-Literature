import streamlit as st
from typing import List, Dict, Any
from dataclasses import dataclass
import time
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from src.config import Config
from src.embeddings.embedding_service import OpenAIEmbeddingService
from src.embeddings.vector_store import VectorStore
from src.retrieval.retriever import DocumentRetriever
from src.llm.llm_client import LLMGenerator
from src.rag_pipeline.pipeline import RAGPipeline

@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    sources: List[Dict[str, Any]] = None

class ChatInterface:
    def __init__(self):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "rag_pipeline" not in st.session_state:
            st.session_state.rag_pipeline = None
    
    def render(self):
        st.header("💬 Chat with AI Papers")
        
        # Initialize RAG pipeline if not already done
        if st.session_state.rag_pipeline is None:
            self._initialize_rag_pipeline()
        
        # Display chat history
        self._display_chat_history()
        
        # Chat input
        self._render_chat_input()
        
        # Sidebar controls
        self._render_sidebar_controls()
    
    def _initialize_rag_pipeline(self):
        with st.spinner("Initializing RAG pipeline..."):
            try:
                # Initialize embedding service
                embedding_service = OpenAIEmbeddingService(
                    model=Config.EMBEDDING_MODEL,
                    api_key=Config.OPENAI_API_KEY
                )
                
                # Initialize vector store
                vector_store = VectorStore(
                    collection_name="ai_papers",
                    persist_directory=Config.VECTOR_DB_PATH
                )
                
                # Initialize retriever
                retriever = DocumentRetriever(vector_store, embedding_service)
                
                # Initialize LLM client
                llm_client = LLMGenerator(
                    model=Config.LLM_MODEL,
                    api_key=Config.OPENAI_API_KEY
                )
                
                # Initialize RAG pipeline
                rag_pipeline = RAGPipeline(retriever, llm_client)
                
                st.session_state.rag_pipeline = rag_pipeline
                st.success("RAG pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize RAG pipeline: {e}")
                st.session_state.rag_pipeline = None
    
    def _display_chat_history(self):
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message.role):
                    st.write(message.content)
                    
                    if message.sources and message.role == "assistant":
                        with st.expander("View Sources"):
                            self._display_sources(message.sources)
    
    def _render_chat_input(self):
        user_input = st.chat_input("Ask a question about AI research papers...")
        
        if user_input:
            # Add user message
            user_message = ChatMessage(
                role="user",
                content=user_input,
                timestamp=time.time()
            )
            st.session_state.chat_history.append(user_message)
            
            # Generate response
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating response..."):
                    response, sources = self._generate_response(user_input)
                    st.write(response)
                    
                    if sources:
                        with st.expander("View Sources"):
                            self._display_sources(sources)
            
            # Add assistant message
            assistant_message = ChatMessage(
                role="assistant",
                content=response,
                timestamp=time.time(),
                sources=sources
            )
            st.session_state.chat_history.append(assistant_message)
    
    def _generate_response(self, query: str) -> tuple:
        if st.session_state.rag_pipeline is None:
            return "RAG pipeline not initialized. Please check your configuration.", []
        
        try:
            # Get settings from session state
            num_sources = getattr(st.session_state, 'num_sources', 5)
            temperature = getattr(st.session_state, 'temperature', 0.1)
            
            # Query the RAG pipeline
            result = st.session_state.rag_pipeline.query(
                user_query=query,
                k=num_sources,
                temperature=temperature
            )
            
            # Extract response and sources from RAGResponse object
            response = result.answer
            retrieved_docs = result.sources
            
            # Format sources for display
            sources = []
            for doc in retrieved_docs:
                metadata = doc.get('metadata', {})
                sources.append({
                    "title": metadata.get('title', doc.get('source_name', 'Unknown Title')),
                    "authors": metadata.get('authors', 'Unknown Author'),
                    "score": doc.get('similarity_score', 0.0),
                    "content": doc.get('content_preview', 'No content available')
                })
            
            return response, sources
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}", []
    
    def _display_sources(self, sources: List[Dict[str, Any]]):
        for i, source in enumerate(sources, 1):
            st.write(f"**Source {i}:** {source.get('title', 'Unknown Title')}")
            st.write(f"**Authors:** {', '.join(source.get('authors', []))}")
            st.write(f"**Relevance Score:** {source.get('score', 0):.2f}")
            st.write(f"**Excerpt:** {source.get('content', '')[:200]}...")
            st.divider()
    
    def _render_sidebar_controls(self):
        with st.sidebar:
            st.subheader("Chat Controls")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            st.subheader("Search Settings")
            num_sources = st.slider("Number of Sources", 1, 10, 5)
            temperature = st.slider("Response Temperature", 0.0, 1.0, 0.1)
            
            # Store settings in session state
            st.session_state.num_sources = num_sources
            st.session_state.temperature = temperature