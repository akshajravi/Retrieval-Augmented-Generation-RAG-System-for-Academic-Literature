import streamlit as st
from typing import List, Dict, Any
from dataclasses import dataclass
import time

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
                # TODO: Initialize actual RAG pipeline components
                # For now, just set a placeholder
                st.session_state.rag_pipeline = "initialized"
                st.success("RAG pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize RAG pipeline: {e}")
    
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
        # TODO: Implement actual RAG pipeline query
        # For now, return a placeholder response
        response = f"I understand you're asking about: '{query}'. The RAG pipeline is not yet fully implemented, but this is where I would search through the indexed papers and provide a comprehensive answer with citations."
        sources = [
            {
                "title": "Example Paper 1",
                "authors": ["Author A", "Author B"],
                "score": 0.85,
                "content": "This is an example source excerpt..."
            }
        ]
        return response, sources
    
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