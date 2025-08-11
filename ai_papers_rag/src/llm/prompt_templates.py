from typing import List, Dict, Any
from string import Template

class PromptTemplates:
    RAG_SYSTEM_PROMPT = """
    You are an AI assistant specialized in analyzing academic papers and research documents.
    You will be provided with context from relevant research papers to answer user questions.
    
    Guidelines:
    - Base your answers on the provided context
    - Cite specific papers when possible
    - If the context doesn't contain enough information, say so
    - Be precise and academic in your responses
    - Include relevant quotes when appropriate
    """
    
    RAG_USER_PROMPT = Template("""
    Context from research papers:
    $context
    
    Question: $query
    
    Please provide a comprehensive answer based on the context above. Include citations where appropriate.
    """)
    
    SUMMARIZATION_PROMPT = Template("""
    Please provide a concise summary of the following research paper excerpt:
    
    $text
    
    Summary should include:
    - Main findings or contributions
    - Key methodologies
    - Important conclusions
    """)
    
    CITATION_PROMPT = Template("""
    Based on the following context from research papers, provide proper citations for the information:
    
    $context
    
    Format citations as: (Author et al., Year) or [Paper Title, Year]
    """)
    
    @classmethod
    def format_rag_prompt(cls, query: str, context: List[str]) -> str:
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        return cls.RAG_USER_PROMPT.substitute(query=query, context=context_text)
    
    @classmethod
    def format_summarization_prompt(cls, text: str) -> str:
        return cls.SUMMARIZATION_PROMPT.substitute(text=text)
    
    @classmethod
    def format_citation_prompt(cls, context: str) -> str:
        return cls.CITATION_PROMPT.substitute(context=context)