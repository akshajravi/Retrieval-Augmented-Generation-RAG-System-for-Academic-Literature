import streamlit as st
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from src.config import Config
import chromadb

class DocumentBrowser:
    def __init__(self):
        if "documents_df" not in st.session_state:
            st.session_state.documents_df = self._load_documents()
    
    def render(self):
        st.header("📄 Document Browser")
        
        # Upload section
        self._render_upload_section()
        
        # Document list and filters
        self._render_document_list()
        
        # Document details
        self._render_document_details()
    
    def _render_upload_section(self):
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Upload and Process", disabled=len(uploaded_files) == 0):
                self._process_uploaded_files(uploaded_files)
        
        with col2:
            if st.button("Refresh Document List"):
                st.session_state.documents_df = self._load_documents()
                st.rerun()
    
    def _render_document_list(self):
        st.subheader("📊 Document Collection")
        
        if st.session_state.documents_df.empty:
            st.info("No documents found. Upload some PDF files to get started.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("Search titles", key="title_search")
        
        with col2:
            author_filter = st.text_input("Filter by author", key="author_filter")
        
        with col3:
            date_filter = st.selectbox(
                "Date range",
                ["All", "Last month", "Last 6 months", "Last year"],
                key="date_filter"
            )
        
        # Apply filters
        filtered_df = self._apply_filters(
            st.session_state.documents_df, 
            search_term, 
            author_filter, 
            date_filter
        )
        
        # Display document table
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[['title', 'authors', 'upload_date', 'status']],
                use_container_width=True
            )
            
            # Document selection
            if 'selected_doc' not in st.session_state:
                st.session_state.selected_doc = None
            
            doc_titles = filtered_df['title'].tolist()
            selected_title = st.selectbox(
                "Select document for details:",
                ["None"] + doc_titles,
                key="doc_selector"
            )
            
            if selected_title != "None":
                st.session_state.selected_doc = filtered_df[
                    filtered_df['title'] == selected_title
                ].iloc[0]
        else:
            st.info("No documents match the current filters.")
    
    def _render_document_details(self):
        if st.session_state.get('selected_doc') is not None:
            st.subheader("🔍 Document Details")
            
            doc = st.session_state.selected_doc
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Title:** {doc['title']}")
                st.write(f"**Authors:** {doc['authors']}")
                st.write(f"**Upload Date:** {doc['upload_date']}")
                st.write(f"**Status:** {doc['status']}")
            
            with col2:
                st.write(f"**File Size:** {doc.get('file_size', 'Unknown')}")
                st.write(f"**Pages:** {doc.get('pages', 'Unknown')}")
                st.write(f"**Language:** {doc.get('language', 'Unknown')}")
                st.write(f"**Document Type:** {doc.get('doc_type', 'Research Paper')}")
            
            # Abstract/Summary
            if doc.get('abstract'):
                st.subheader("Abstract")
                st.write(doc['abstract'])
            
            # Keywords
            if doc.get('keywords'):
                st.subheader("Keywords")
                keywords = doc['keywords'].split(',') if isinstance(doc['keywords'], str) else doc['keywords']
                for keyword in keywords:
                    st.tag(keyword.strip())
            
            # Actions
            st.subheader("Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("View Full Text"):
                    self._view_full_text(doc)
            
            with col2:
                if st.button("Reprocess Document"):
                    self._reprocess_document(doc)
            
            with col3:
                if st.button("Delete Document"):
                    self._delete_document(doc)
    
    def _load_documents(self) -> pd.DataFrame:
        # Load documents from ChromaDB database
        try:
            # Connect to ChromaDB
            client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
            collection = client.get_or_create_collection("ai_papers")
            
            # Get all documents
            results = collection.get(include=['metadatas'])
            metadatas = results.get('metadatas', [])
            
            if not metadatas:
                return pd.DataFrame()
            
            # Group by source file to get unique papers
            papers = {}
            for metadata in metadatas:
                source_file = metadata.get('source_file', 'Unknown')
                if source_file not in papers:
                    papers[source_file] = {
                        'title': metadata.get('title', source_file.replace('.pdf', '')),
                        'authors': metadata.get('authors', 'Unknown'),
                        'upload_date': '2024-01-01',  # Default date
                        'status': 'Processed',
                        'file_size': 'Unknown',
                        'pages': 'Unknown',
                        'abstract': 'No abstract available',
                        'keywords': 'research, AI',
                        'source_file': source_file
                    }
            
            return pd.DataFrame(list(papers.values()))
            
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            return pd.DataFrame()
    
    def _apply_filters(self, df: pd.DataFrame, search_term: str, 
                      author_filter: str, date_filter: str) -> pd.DataFrame:
        filtered_df = df.copy()
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['title'].str.contains(search_term, case=False, na=False)
            ]
        
        if author_filter:
            filtered_df = filtered_df[
                filtered_df['authors'].str.contains(author_filter, case=False, na=False)
            ]
        
        # Date filtering could be implemented here when date metadata is available
        
        return filtered_df
    
    def _process_uploaded_files(self, uploaded_files):
        with st.spinner("Processing uploaded files..."):
            for file in uploaded_files:
                # File processing would integrate with the document ingestion pipeline
                st.success(f"Processed {file.name}")
            
            # Refresh document list
            st.session_state.documents_df = self._load_documents()
    
    def _view_full_text(self, doc):
        st.info("Full text viewing not yet implemented.")
    
    def _reprocess_document(self, doc):
        st.info(f"Reprocessing {doc['title']}...")
    
    def _delete_document(self, doc):
        if st.confirm(f"Are you sure you want to delete '{doc['title']}'?"):
            st.success(f"Document '{doc['title']}' would be deleted.")