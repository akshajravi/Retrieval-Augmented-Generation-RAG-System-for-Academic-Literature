import streamlit as st
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class SourceDisplay:
    def __init__(self):
        if "search_results" not in st.session_state:
            st.session_state.search_results = []
    
    def render(self):
        st.header("🔍 Source Display & Analysis")
        
        # Search interface
        self._render_search_interface()
        
        # Results display
        if st.session_state.search_results:
            self._render_search_results()
            self._render_analysis_dashboard()
        else:
            st.info("No search results to display. Use the search interface above to find relevant sources.")
    
    def _render_search_interface(self):
        st.subheader("🔎 Search Interface")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter your research query:",
                placeholder="e.g., transformer attention mechanisms"
            )
        
        with col2:
            search_button = st.button("Search", type="primary")
        
        # Advanced search options
        with st.expander("⚙️ Advanced Search Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_results = st.slider("Number of results", 1, 20, 10)
                similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7)
            
            with col2:
                date_range = st.date_input(
                    "Publication date range",
                    value=None,
                    help="Filter by publication date"
                )
                author_filter = st.text_input("Filter by author")
            
            with col3:
                search_type = st.selectbox(
                    "Search type",
                    ["Semantic", "Keyword", "Hybrid"]
                )
                sort_by = st.selectbox(
                    "Sort by",
                    ["Relevance", "Date", "Citations"]
                )
        
        if search_button and search_query:
            self._perform_search(search_query, num_results, similarity_threshold)
    
    def _render_search_results(self):
        st.subheader(f"📄 Search Results ({len(st.session_state.search_results)} found)")
        
        # Results overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = sum(r['score'] for r in st.session_state.search_results) / len(st.session_state.search_results)
            st.metric("Average Relevance Score", f"{avg_score:.2f}")
        
        with col2:
            unique_papers = len(set(r['paper_title'] for r in st.session_state.search_results))
            st.metric("Unique Papers", unique_papers)
        
        with col3:
            total_citations = sum(r.get('citations', 0) for r in st.session_state.search_results)
            st.metric("Total Citations", total_citations)
        
        # Individual results
        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{i+1}. {result['paper_title']}**")
                    st.write(f"*Authors:* {result['authors']}")
                    st.write(f"*Excerpt:* {result['content'][:200]}...")
                    
                    # Tags
                    if result.get('keywords'):
                        for keyword in result['keywords']:
                            st.tag(keyword)
                
                with col2:
                    st.metric("Relevance", f"{result['score']:.2f}")
                    if result.get('citations'):
                        st.metric("Citations", result['citations'])
                    
                    if st.button(f"View Details", key=f"detail_{i}"):
                        self._show_result_details(result)
                
                st.divider()
    
    def _render_analysis_dashboard(self):
        st.subheader("📊 Analysis Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["Score Distribution", "Paper Analysis", "Keyword Cloud"])
        
        with tab1:
            self._render_score_distribution()
        
        with tab2:
            self._render_paper_analysis()
        
        with tab3:
            self._render_keyword_analysis()
    
    def _render_score_distribution(self):
        scores = [r['score'] for r in st.session_state.search_results]
        
        fig = px.histogram(
            x=scores,
            bins=10,
            title="Distribution of Relevance Scores",
            labels={'x': 'Relevance Score', 'y': 'Count'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Score statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Score", f"{min(scores):.2f}")
        with col2:
            st.metric("Max Score", f"{max(scores):.2f}")
        with col3:
            st.metric("Mean Score", f"{sum(scores)/len(scores):.2f}")
        with col4:
            st.metric("Std Dev", f"{pd.Series(scores).std():.2f}")
    
    def _render_paper_analysis(self):
        # Group results by paper
        paper_counts = {}
        for result in st.session_state.search_results:
            paper = result['paper_title']
            if paper not in paper_counts:
                paper_counts[paper] = {
                    'count': 0,
                    'max_score': 0,
                    'avg_score': 0,
                    'authors': result['authors']
                }
            paper_counts[paper]['count'] += 1
            paper_counts[paper]['max_score'] = max(paper_counts[paper]['max_score'], result['score'])
        
        # Calculate average scores
        for paper in paper_counts:
            scores = [r['score'] for r in st.session_state.search_results if r['paper_title'] == paper]
            paper_counts[paper]['avg_score'] = sum(scores) / len(scores)
        
        # Create DataFrame for visualization
        df = pd.DataFrame([
            {
                'Paper': paper[:50] + '...' if len(paper) > 50 else paper,
                'Chunks Found': data['count'],
                'Max Score': data['max_score'],
                'Avg Score': data['avg_score']
            }
            for paper, data in paper_counts.items()
        ])
        
        fig = px.scatter(
            df,
            x='Chunks Found',
            y='Max Score',
            size='Avg Score',
            hover_data=['Paper'],
            title="Paper Relevance Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top papers table
        st.write("**Top Papers by Relevance:**")
        top_papers = sorted(paper_counts.items(), key=lambda x: x[1]['max_score'], reverse=True)[:5]
        
        for i, (paper, data) in enumerate(top_papers, 1):
            st.write(f"{i}. **{paper}** - Max Score: {data['max_score']:.2f}, Chunks: {data['count']}")
    
    def _render_keyword_analysis(self):
        # Collect all keywords
        all_keywords = []
        for result in st.session_state.search_results:
            if result.get('keywords'):
                all_keywords.extend(result['keywords'])
        
        if not all_keywords:
            st.info("No keywords available for analysis.")
            return
        
        # Count keyword frequency
        keyword_counts = pd.Series(all_keywords).value_counts()
        
        # Create bar chart
        fig = px.bar(
            x=keyword_counts.values[:15],
            y=keyword_counts.index[:15],
            orientation='h',
            title="Top 15 Keywords in Search Results",
            labels={'x': 'Frequency', 'y': 'Keywords'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Unique Keywords", len(keyword_counts))
            st.metric("Total Keyword Mentions", len(all_keywords))
        
        with col2:
            st.metric("Most Common Keyword", keyword_counts.index[0])
            st.metric("Frequency", keyword_counts.iloc[0])
    
    def _perform_search(self, query: str, num_results: int, threshold: float):
        with st.spinner("Searching through documents..."):
            # Search functionality placeholder
            # For now, return mock results
            mock_results = self._generate_mock_results(query, num_results)
            st.session_state.search_results = mock_results
            st.success(f"Found {len(mock_results)} relevant sources!")
    
    def _generate_mock_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        import random
        
        mock_papers = [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "GPT-3: Language Models are Few-Shot Learners",
            "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
        ]
        
        mock_authors = [
            "Vaswani et al.",
            "Devlin et al.",
            "Brown et al.",
            "Dai et al.",
            "Liu et al."
        ]
        
        mock_keywords = [
            ["attention", "transformer", "neural networks"],
            ["BERT", "language model", "pre-training"],
            ["GPT", "few-shot learning", "language generation"],
            ["transformer", "long context", "attention"],
            ["BERT", "robustness", "optimization"]
        ]
        
        results = []
        for i in range(min(num_results, 10)):
            paper_idx = i % len(mock_papers)
            results.append({
                'paper_title': mock_papers[paper_idx],
                'authors': mock_authors[paper_idx],
                'content': f"This section discusses {query} in the context of {mock_papers[paper_idx]}. The approach demonstrates significant improvements over previous methods...",
                'score': random.uniform(0.6, 0.95),
                'citations': random.randint(100, 5000),
                'keywords': mock_keywords[paper_idx]
            })
        
        return results
    
    def _show_result_details(self, result: Dict[str, Any]):
        st.modal(f"Details: {result['paper_title']}")
        with st.modal:
            st.write(f"**Paper:** {result['paper_title']}")
            st.write(f"**Authors:** {result['authors']}")
            st.write(f"**Relevance Score:** {result['score']:.3f}")
            st.write(f"**Citations:** {result.get('citations', 'Unknown')}")
            
            st.subheader("Full Excerpt")
            st.write(result['content'])
            
            if result.get('keywords'):
                st.subheader("Keywords")
                for keyword in result['keywords']:
                    st.tag(keyword)