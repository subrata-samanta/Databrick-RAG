"""
Module for vector store setup and management.
"""

import logging
from langchain_databricks import DatabricksVectorSearch
from langchain_databricks.embeddings import DatabricksEmbeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Class for managing vector store setup"""
    
    def __init__(self, config):
        """
        Initialize vector store manager
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def setup_vector_store(self):
        """
        Set up vector store for retrieval
        
        Returns:
            Retriever object
        """
        try:
            # Initialize embedding model
            embeddings = DatabricksEmbeddings(
                endpoint=self.config.get_embedding_model()
            )
            
            # Initialize vector store
            vector_store = DatabricksVectorSearch(
                embedding_function=embeddings,
                catalog=self.config.get_catalog_name(),
                schema=self.config.get_schema_name(),
                table=self.config.base_config["vector_search"]["embeddings_table_name"],
                vector_column=self.config.base_config["vector_search"]["vector_column_name"],
                text_column=self.config.base_config["vector_search"]["text_column_name"]
            )
            
            # Get retriever
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 5}  # Return top 5 most relevant documents
            )
            
            logger.info("Vector store and retriever set up successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to set up vector store: {str(e)}")
            raise
