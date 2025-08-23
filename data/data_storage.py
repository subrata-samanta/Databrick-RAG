"""
Module for managing data storage in Databricks volumes and tables.
"""

import logging
from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)

class DataStorage:
    """Class for managing data storage in Databricks"""
    
    def __init__(self, spark: SparkSession, config):
        """
        Initialize data storage manager
        
        Args:
            spark: SparkSession
            config: Configuration object
        """
        self.spark = spark
        self.config = config
        self.catalog_name = config.get_catalog_name()
        self.schema_name = config.get_schema_name()
    
    def setup_volumes(self) -> str:
        """
        Set up Volumes for unstructured data storage
        
        Returns:
            Volume path
        """
        try:
            volume_path = self.config.get_volume_path()
            
            # Create catalog if it doesn't exist
            self.spark.sql(f"CREATE CATALOG IF NOT EXISTS {self.catalog_name}")
            
            # Create schema if it doesn't exist
            self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog_name}.{self.schema_name}")
            
            # Create volume if it doesn't exist
            self.spark.sql(f"CREATE VOLUME IF NOT EXISTS {self.catalog_name}.{self.schema_name}.documents")
            
            logger.info(f"Volumes setup completed at {volume_path}")
            return volume_path
        except Exception as e:
            logger.error(f"Failed to set up volumes: {str(e)}")
            raise
    
    def save_documents_to_volume(self, docs_df: DataFrame) -> str:
        """
        Save documents to volume and register as table
        
        Args:
            docs_df: DataFrame containing documents
            
        Returns:
            Full table name
        """
        try:
            volume_path = self.setup_volumes()
            
            # Write documents to Delta table in the volume
            docs_df.write.format("delta").mode("overwrite").save(volume_path)
            
            # Create UC table on top of the volume
            table_name = self.config.get_full_table_name(self.config.base_config["vector_search"]["source_table_name"])
            self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {table_name}
            USING DELTA
            LOCATION '{volume_path}'
            """)
            
            logger.info(f"Documents saved to volume and registered as table {table_name}")
            return table_name
        except Exception as e:
            logger.error(f"Failed to save documents to volume: {str(e)}")
            raise
    
    def save_structured_data_to_tables(self, customers_df: DataFrame, products_df: DataFrame, orders_df: DataFrame):
        """
        Save structured data to tables
        
        Args:
            customers_df: DataFrame with customer data
            products_df: DataFrame with product data
            orders_df: DataFrame with order data
        """
        try:
            # Create tables
            customers_df.write.format("delta").mode("overwrite").saveAsTable(
                self.config.get_full_table_name("customers")
            )
            products_df.write.format("delta").mode("overwrite").saveAsTable(
                self.config.get_full_table_name("products")
            )
            orders_df.write.format("delta").mode("overwrite").saveAsTable(
                self.config.get_full_table_name("orders")
            )
            
            logger.info(f"Structured data saved to tables in {self.catalog_name}.{self.schema_name}")
        except Exception as e:
            logger.error(f"Failed to save structured data to tables: {str(e)}")
            raise
    
    def setup_inference_tables(self) -> str:
        """
        Set up inference tables for logging
        
        Returns:
            Inference table name
        """
        try:
            # Create inference tables for logging
            inference_table = self.config.get_full_table_name("inference_logs")
            
            self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {inference_table} (
                request_id STRING,
                timestamp TIMESTAMP,
                query STRING,
                response STRING,
                latency_ms DOUBLE,
                tokens_input INT,
                tokens_output INT,
                environment STRING,
                endpoint_name STRING,
                model_name STRING,
                user_id STRING,
                metadata MAP<STRING, STRING>
            )
            USING delta
            PARTITIONED BY (date_day STRING)
            """)
            
            logger.info(f"Created inference table: {inference_table}")
            return inference_table
        except Exception as e:
            logger.error(f"Failed to setup inference tables: {str(e)}")
            raise
