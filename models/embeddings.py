"""
Module for managing embedding models and vector search.
"""

import logging
from pyspark.sql import SparkSession
from databricks.vector import VectorSearchClient
from databricks.sdk.service import workspace
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Class for managing embeddings and vector search"""
    
    def __init__(self, spark: SparkSession, config, workspace_client: WorkspaceClient = None):
        """
        Initialize embedding manager
        
        Args:
            spark: SparkSession
            config: Configuration object
            workspace_client: Optional WorkspaceClient
        """
        self.spark = spark
        self.config = config
        self.workspace_client = workspace_client or WorkspaceClient()
        self.vector_client = VectorSearchClient(workspace_client=self.workspace_client)
    
    def create_embeddings_and_vector_index(self) -> tuple:
        """
        Generate embeddings for documents and create vector index
        
        Returns:
            Tuple of (embeddings_table_name, vector_index_name)
        """
        try:
            catalog_name = self.config.get_catalog_name()
            schema_name = self.config.get_schema_name()
            source_table = self.config.base_config["vector_search"]["source_table_name"]
            embeddings_table = self.config.base_config["vector_search"]["embeddings_table_name"]
            vector_column = self.config.base_config["vector_search"]["vector_column_name"]
            text_column = self.config.base_config["vector_search"]["text_column_name"]
            index_name = self.config.base_config["uc_config"]["vector_index_name"]
            
            # Process documents and create chunks for embedding
            query = f"""
            WITH chunks AS (
                SELECT 
                    id,
                    title,
                    EXPLODE(SPLIT_TEXT(content, 'paragraph', 1000)) as content_chunk,
                    metadata
                FROM {catalog_name}.{schema_name}.{source_table}
            )
            SELECT 
                CONCAT(id, '-chunk-', MONOTONICALLY_INCREASING_ID()) as chunk_id,
                id as document_id,
                title,
                content_chunk as {text_column},
                metadata
            FROM chunks
            """
            
            # Create chunks table
            self.spark.sql(query).write.format("delta").mode("overwrite").saveAsTable(
                f"{catalog_name}.{schema_name}.document_chunks"
            )
            
            # Generate embeddings using Databricks embedding model
            embedding_endpoint_name = self.config.get_embedding_model()
            
            embedding_query = f"""
            SELECT 
                chunk_id,
                document_id,
                title,
                {text_column},
                metadata,
                databricks_embedding('{embedding_endpoint_name}', {text_column}) as {vector_column}
            FROM {catalog_name}.{schema_name}.document_chunks
            """
            
            # Create embeddings table
            self.spark.sql(embedding_query).write.format("delta").mode("overwrite").saveAsTable(
                f"{catalog_name}.{schema_name}.{embeddings_table}"
            )
            
            # Create vector index
            self.vector_client.create_index(
                source_table_name=f"{catalog_name}.{schema_name}.{embeddings_table}",
                index_name=index_name,
                primary_key="chunk_id",
                vector_column_name=vector_column,
                embedding_dimension=self.config.base_config["vector_search"]["dimension"],
                metric_type=self.config.base_config["vector_search"]["metric"],
                sync=True
            )
            
            embeddings_table_name = f"{catalog_name}.{schema_name}.{embeddings_table}"
            logger.info(f"Created vector index {index_name} on table {embeddings_table_name}")
            return embeddings_table_name, index_name
            
        except Exception as e:
            logger.error(f"Failed to create embeddings and vector index: {str(e)}")
            raise
