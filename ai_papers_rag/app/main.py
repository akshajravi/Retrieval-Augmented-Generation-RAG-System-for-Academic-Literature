import streamlit as st
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from components.chat_interface import ChatInterface
from components.document_browser import DocumentBrowser
from components.source_display import SourceDisplay
from utils.session_manager import SessionManager

def main():
    st.set_page_config(
        page_title="AI Papers RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 AI Papers RAG System")
    st.markdown("Query and analyze AI research papers using advanced retrieval-augmented generation.")
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page:",
            ["💬 Chat Interface", "📄 Document Browser", "🔍 Source Display", "⚙️ Settings"]
        )
    
    # Main content based on selected page
    if page == "💬 Chat Interface":
        chat_interface = ChatInterface()
        chat_interface.render()
    
    elif page == "📄 Document Browser":
        document_browser = DocumentBrowser()
        document_browser.render()
    
    elif page == "🔍 Source Display":
        source_display = SourceDisplay()
        source_display.render()
    
    elif page == "⚙️ Settings":
        render_settings_page()

def render_settings_page():
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        st.text_input("LLM Model", value=Config.LLM_MODEL, disabled=True)
        st.text_input("Embedding Model", value=Config.EMBEDDING_MODEL, disabled=True)
        st.slider("Temperature", 0.0, 2.0, Config.LLM_TEMPERATURE, disabled=True)
    
    with col2:
        st.subheader("Retrieval Configuration")
        st.number_input("Number of Results (K)", value=Config.RETRIEVAL_K, disabled=True)
        st.number_input("Chunk Size", value=Config.CHUNK_SIZE, disabled=True)
        st.number_input("Chunk Overlap", value=Config.CHUNK_OVERLAP, disabled=True)
    
    st.info("Configuration is read from environment variables and config files.")

if __name__ == "__main__":
    main()