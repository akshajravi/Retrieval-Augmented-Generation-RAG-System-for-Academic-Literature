from typing import List, Dict, Any, Optional
from string import Template

class PromptTemplates:
    """Centralized prompt template management for academic RAG system."""
    
    RAG_SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing academic papers and research documents.
You will be provided with context from relevant research papers to answer user questions.

Guidelines:
- Base your answers strictly on the provided context
- When referencing information, use [SOURCE_X] format where X matches the source number
- If the context doesn't contain enough information, clearly state this
- Be precise and academic in your responses
- Synthesize information from multiple sources when available
- Do not make claims beyond what the context supports"""
    
    RAG_USER_PROMPT = Template("""Context from research papers:
$context

Question: $query

Please provide a comprehensive answer based on the context above. Use [SOURCE_X] format to cite sources where X matches the source number in the context.""")
    
    SUMMARIZATION_PROMPT = Template("""Please provide a concise summary of the following research paper excerpt:

$text

Summary should include:
- Main findings or contributions
- Key methodologies used
- Important conclusions or implications
- Significance to the field

Keep the summary focused and under 200 words.""")
    
    COMPARISON_PROMPT = Template("""Based on the following context from multiple research papers, compare and contrast the concepts, methods, or findings mentioned in the question:

Context:
$context

Question: $query

Please provide a structured comparison that:
- Highlights similarities and differences
- Cites specific papers using [SOURCE_X] format
- Discusses advantages and limitations where mentioned
- Synthesizes insights from multiple sources""")
    
    DEFINITION_PROMPT = Template("""Based on the following context from research papers, provide a clear definition and explanation:

Context:
$context

Question: $query

Please provide:
- A clear, precise definition
- Technical details from the papers
- Examples or applications mentioned
- Different perspectives if multiple papers discuss the topic
- Use [SOURCE_X] format for citations""")
    
    METHODOLOGY_PROMPT = Template("""Based on the following context, explain the methodology, approach, or technical process:

Context:
$context

Question: $query

Please explain:
- Step-by-step process if available
- Technical details and parameters
- Algorithms or formulas mentioned
- Implementation considerations
- Evaluation methods used
- Use [SOURCE_X] format for citations""")
    
    CITATION_PROMPT = Template("""Based on the following context from research papers, provide proper citations for the information:

$context

Format citations as: [Paper Title] or [Author et al.] depending on what information is available in the context.""")
    
    # Intent-specific prompts for better responses
    INTENT_PROMPTS = {
        "definition": DEFINITION_PROMPT,
        "comparison": COMPARISON_PROMPT,
        "methodology": METHODOLOGY_PROMPT,
        "general": RAG_USER_PROMPT
    }
    
    @classmethod
    def format_rag_prompt(cls, query: str, context: List[str], intent: str = "general") -> str:
        """Format RAG prompt based on query intent."""
        # Format context with source numbering
        context_text = cls._format_context_with_sources(context)
        
        # Select appropriate prompt template
        template = cls.INTENT_PROMPTS.get(intent, cls.RAG_USER_PROMPT)
        
        return template.substitute(query=query, context=context_text)
    
    @classmethod
    def format_rag_prompt_with_search_results(cls, query: str, search_results: List[Any], 
                                            intent: str = "general") -> str:
        """Format RAG prompt using search results with metadata."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            # Extract source information
            source_file = result.metadata.get('source_file', 'Unknown Source')
            source_name = cls.clean_source_name(source_file)
            
            # Format each context chunk with source
            context_parts.append(f"[SOURCE_{i}: {source_name}]\n{result.content.strip()}")
        
        context_text = "\n\n".join(context_parts)
        
        # Select appropriate prompt template
        template = cls.INTENT_PROMPTS.get(intent, cls.RAG_USER_PROMPT)
        
        return template.substitute(query=query, context=context_text)
    
    @classmethod
    def format_summarization_prompt(cls, text: str) -> str:
        """Format text summarization prompt."""
        return cls.SUMMARIZATION_PROMPT.substitute(text=text)
    
    @classmethod
    def format_citation_prompt(cls, context: str) -> str:
        """Format citation extraction prompt."""
        return cls.CITATION_PROMPT.substitute(context=context)
    
    @classmethod
    def create_system_message(cls) -> Dict[str, str]:
        """Create system message for chat-based models."""
        return {"role": "system", "content": cls.RAG_SYSTEM_PROMPT}
    
    @classmethod
    def create_user_message(cls, query: str, context: List[str], intent: str = "general") -> Dict[str, str]:
        """Create user message for chat-based models."""
        prompt = cls.format_rag_prompt(query, context, intent)
        return {"role": "user", "content": prompt}
    
    @classmethod
    def create_chat_messages(cls, query: str, context: List[str], intent: str = "general") -> List[Dict[str, str]]:
        """Create complete message list for chat models."""
        return [
            cls.create_system_message(),
            cls.create_user_message(query, context, intent)
        ]
    
    @classmethod
    def _format_context_with_sources(cls, context: List[str]) -> str:
        """Format context list with source numbering."""
        if not context:
            return "No relevant context found."
        
        formatted_chunks = []
        for i, chunk in enumerate(context, 1):
            formatted_chunks.append(f"[SOURCE_{i}]\n{chunk.strip()}")
        
        return "\n\n".join(formatted_chunks)
    
    @classmethod
    def clean_source_name(cls, source_file: str) -> str:
        """Convert filename to readable source name."""
        name = source_file.replace('.pdf', '')
        name = name.replace('_', ' ').title()
        return name
    
    @classmethod
    def get_available_intents(cls) -> List[str]:
        """Get list of available intent types."""
        return list(cls.INTENT_PROMPTS.keys())
    
    @classmethod
    def validate_intent(cls, intent: str) -> str:
        """Validate and return intent, defaulting to general if invalid."""
        return intent if intent in cls.INTENT_PROMPTS else "general"


    
    