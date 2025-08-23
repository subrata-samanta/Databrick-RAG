"""
Module for SQL agent with text-to-SQL capabilities.
"""

import logging
from langchain_databricks.chat_models import ChatDatabricks
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

logger = logging.getLogger(__name__)

class SQLAgentManager:
    """Class for managing SQL agent with text-to-SQL capabilities"""
    
    def __init__(self, config):
        """
        Initialize SQL agent manager
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def create_sql_agent(self):
        """
        Create SQL agent with Genie for text-to-SQL
        
        Returns:
            SQL agent object
        """
        try:
            # Get warehouse ID from config
            warehouse_id = self.config.get_warehouse_id()
            
            # Get catalog and schema
            catalog_name = self.config.get_catalog_name()
            schema_name = self.config.get_schema_name()
            
            # Set up connection string
            connection_string = f"databricks://sql/warehouse/{warehouse_id};catalog={catalog_name};schema={schema_name}"
            
            # Create SQL Database connection
            db = SQLDatabase.from_uri(connection_string)
            
            # Initialize LLM for text-to-SQL
            llm = ChatDatabricks(endpoint=self.config.get_llm_model())
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            # Create SQL agent
            sql_agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="zero-shot-react-description",
                handle_parsing_errors=True
            )
            
            logger.info("Created SQL Agent with Genie capabilities")
            return sql_agent
            
        except Exception as e:
            logger.error(f"Failed to create SQL agent: {str(e)}")
            raise
