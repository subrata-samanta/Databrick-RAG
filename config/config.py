"""
Configuration module for the RAG pipeline.
Contains all configurable parameters for the application.
"""

import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for RAG pipeline"""
    
    def __init__(self, environment="dev"):
        """
        Initialize configuration with specified environment
        
        Args:
            environment: The deployment environment (dev, qa, or prod)
        """
        self.environment = environment
        self._load_config()
        
    def _load_config(self):
        """Load configuration based on environment"""
        # Base configuration
        self.base_config = {
            # Service Principal for Mosaic Resources
            "service_principal": {
                "client_id": os.environ.get("SP_CLIENT_ID", "YOUR_SP_CLIENT_ID"),
                "client_secret": os.environ.get("SP_CLIENT_SECRET", "YOUR_SP_CLIENT_SECRET"),
                "tenant_id": os.environ.get("SP_TENANT_ID", "YOUR_SP_TENANT_ID")
            },
            
            # Unity Catalog configuration
            "uc_config": {
                "catalog_name": "rag_catalog",
                "schema_name": "rag_schema",
                "model_name": "rag_assistant",
                "vector_index_name": "document_embeddings_index"
            },
            
            # Vector search configuration
            "vector_search": {
                "source_table_name": "documents",
                "embeddings_table_name": "document_embeddings",
                "vector_column_name": "embedding",
                "text_column_name": "content",
                "dimension": 1536,  # For OpenAI embeddings
                "metric": "cosine"
            },
            
            # Model configuration
            "model": {
                "embedding_model": "databricks-bge-large-en",
                "llm_model": "databricks-llama-3-70b-instruct",
                "dev_endpoint_name": "rag-assistant-dev",
                "qa_endpoint_name": "rag-assistant-qa",
                "prod_endpoint_name": "rag-assistant-prod",
                "model_version": "1"
            },
            
            # Infrastructure settings
            "infrastructure": {
                "dev": {
                    "warehouse_id": os.environ.get("DEV_WAREHOUSE_ID", "dev_warehouse_id"),
                    "scale_to_zero": True,
                    "min_scale": 1,
                    "max_scale": 2
                },
                "qa": {
                    "warehouse_id": os.environ.get("QA_WAREHOUSE_ID", "qa_warehouse_id"),
                    "scale_to_zero": True,
                    "min_scale": 1,
                    "max_scale": 4
                },
                "prod": {
                    "warehouse_id": os.environ.get("PROD_WAREHOUSE_ID", "prod_warehouse_id"),
                    "scale_to_zero": False,
                    "provisioned_throughput": 10,
                    "min_scale": 2,
                    "max_scale": 8
                }
            },
            
            # Storage configuration
            "storage": {
                "volume_path": "/Volumes/rag_catalog/rag_schema/documents",
                "data_format": "delta"
            },
            
            # Monitoring configuration
            "monitoring": {
                "dashboard_id": "rag_dashboard",
                "metrics_table": "rag_metrics",
                "log_level": "INFO"
            }
        }
        
        # Set active configuration based on environment
        self.active_config = self.base_config["infrastructure"][self.environment]
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def get_endpoint_name(self):
        """Get the appropriate endpoint name for the current environment"""
        return self.base_config["model"][f"{self.environment}_endpoint_name"]
    
    def get_warehouse_id(self):
        """Get the warehouse ID for the current environment"""
        return self.active_config["warehouse_id"]
    
    def get_catalog_name(self):
        """Get the Unity Catalog name"""
        return self.base_config["uc_config"]["catalog_name"]
    
    def get_schema_name(self):
        """Get the Unity Catalog schema name"""
        return self.base_config["uc_config"]["schema_name"]
    
    def get_full_table_name(self, table_name):
        """Get fully qualified table name with catalog and schema"""
        return f"{self.get_catalog_name()}.{self.get_schema_name()}.{table_name}"
    
    def get_volume_path(self):
        """Get the volume path for document storage"""
        return self.base_config["storage"]["volume_path"]
    
    def get_embedding_model(self):
        """Get the embedding model name"""
        return self.base_config["model"]["embedding_model"]
    
    def get_llm_model(self):
        """Get the LLM model name"""
        return self.base_config["model"]["llm_model"]

# Create default config instance
config = Config()
