"""
Module for RAG chain implementation with LangChain.
"""

import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_databricks.chat_models import ChatDatabricks

logger = logging.getLogger(__name__)

class RAGChainManager:
    """Class for managing RAG chain implementation"""
    
    def __init__(self, config, retriever):
        """
        Initialize RAG chain manager
        
        Args:
            config: Configuration object
            retriever: Document retriever
        """
        self.config = config
        self.retriever = retriever
    
    def create_rag_chain(self):
        """
        Create the RAG chain
        
        Returns:
            RAG chain object
        """
        try:
            # Initialize LLM
            llm = ChatDatabricks(endpoint=self.config.get_llm_model())
            
            # Define RAG prompt template
            template = """You are a helpful AI assistant. Use the following context to answer the question.
            If you don't know the answer, just say you don't know. Don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create the RAG chain
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",  # Simple document concatenation
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True  # Include source documents in response
            )
            
            logger.info("Created RAG chain with LangChain components")
            return rag_chain
            
        except Exception as e:
            logger.error(f"Failed to create RAG chain: {str(e)}")
            raise
