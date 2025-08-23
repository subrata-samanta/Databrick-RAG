"""
Module for query rewriting to improve retrieval effectiveness.
"""

import logging
from typing import Dict, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_databricks.chat_models import ChatDatabricks

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Class for rewriting queries to improve retrieval effectiveness"""
    
    def __init__(self, config, llm=None):
        """
        Initialize query rewriter
        
        Args:
            config: Configuration object
            llm: Optional LLM instance
        """
        self.config = config
        self.llm = llm or ChatDatabricks(endpoint=config.get_llm_model())
        self.setup_rewriter()
    
    def setup_rewriter(self):
        """Set up the query rewriting chain"""
        rewrite_template = """You are an expert query rewriter for a retrieval system.
Your task is to rewrite the user's query to make it more effective for document retrieval.
You should:
1. Expand abbreviations and acronyms
2. Add synonyms for key terms (separated by OR)
3. Remove unnecessary words
4. Clarify ambiguous terms
5. Include key entities and concepts

Original query: {query}
Context (optional): {context}

Rewritten query:"""
        
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=rewrite_template
        )
        
        self.query_rewriter_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=self.config.base_config.get("verbose", False)
        )
        
        logger.info("Query rewriter initialized")
    
    def rewrite_query(self, query: str, context: Optional[str] = None) -> Dict:
        """
        Rewrite the query for better retrieval
        
        Args:
            query: Original user query
            context: Optional context from conversation history
            
        Returns:
            Dictionary with original and rewritten query
        """
        try:
            # If query is very short, don't rewrite
            if len(query.split()) <= 2:
                logger.info("Query too short for rewriting, using original")
                return {
                    "original_query": query,
                    "rewritten_query": query,
                    "rewritten": False
                }
            
            # If context is None, use empty string
            context = context or ""
            
            # Rewrite the query
            rewritten = self.query_rewriter_chain.run(
                query=query,
                context=context
            )
            
            # Log the rewriting
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            
            return {
                "original_query": query,
                "rewritten_query": rewritten.strip(),
                "rewritten": True
            }
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            # Fall back to original query on error
            return {
                "original_query": query,
                "rewritten_query": query,
                "rewritten": False,
                "error": str(e)
            }
    
    def generate_query_variants(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple variants of a query to improve recall
        
        Args:
            query: Original user query
            num_variants: Number of variants to generate
            
        Returns:
            List of query variants
        """
        variants_template = f"""Create {num_variants} different versions of this search query.
Each version should focus on different aspects or use different terms but retain the same intent.
Format the output as a numbered list.

Original query: {{query}}

Variants:"""
        
        variants_prompt = PromptTemplate(
            input_variables=["query"],
            template=variants_template
        )
        
        variants_chain = LLMChain(
            llm=self.llm,
            prompt=variants_prompt
        )
        
        try:
            result = variants_chain.run(query=query)
            
            # Parse the numbered list
            lines = result.strip().split("\n")
            variants = []
            for line in lines:
                if line.strip() and any(c.isdigit() for c in line):
                    # Extract the text after any numbering like "1." or "1)"
                    text = line.split(".", 1)[-1].split(")", 1)[-1].strip()
                    if text:
                        variants.append(text)
            
            # If parsing failed, just split by newlines
            if not variants:
                variants = [line.strip() for line in lines if line.strip()]
            
            # Ensure we don't exceed requested number and include original
            variants = variants[:num_variants-1]
            variants.insert(0, query)
            
            return variants
            
        except Exception as e:
            logger.error(f"Error generating query variants: {str(e)}")
            return [query]  # Return just the original query on error
