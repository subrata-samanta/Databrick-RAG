"""
Module for data loading utilities.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading data from various sources"""
    
    def __init__(self, spark: SparkSession, config):
        """
        Initialize data loader
        
        Args:
            spark: SparkSession
            config: Configuration object
        """
        self.spark = spark
        self.config = config
    
    def load_table(self, table_name: str) -> DataFrame:
        """
        Load data from a Delta table
        
        Args:
            table_name: Table name (without catalog/schema)
            
        Returns:
            Spark DataFrame with table data
        """
        try:
            full_table_name = self.config.get_full_table_name(table_name)
            logger.info(f"Loading data from table: {full_table_name}")
            
            df = self.spark.table(full_table_name)
            logger.info(f"Loaded {df.count()} rows from {full_table_name}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data from table {table_name}: {str(e)}")
            raise
    
    def load_from_volume(self, volume_path: Optional[str] = None, file_format: str = "delta") -> DataFrame:
        """
        Load data from a volume
        
        Args:
            volume_path: Volume path (if None, use the default from config)
            file_format: File format (delta, parquet, csv, etc.)
            
        Returns:
            Spark DataFrame with data from the volume
        """
        try:
            path = volume_path or self.config.get_volume_path()
            logger.info(f"Loading data from volume: {path} (format: {file_format})")
            
            df = self.spark.read.format(file_format).load(path)
            logger.info(f"Loaded {df.count()} rows from {path}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data from volume {volume_path}: {str(e)}")
            raise
    
    def load_sql_query(self, query: str, warehouse_id: Optional[str] = None) -> DataFrame:
        """
        Load data using a SQL query through serverless warehouse
        
        Args:
            query: SQL query to execute
            warehouse_id: Optional warehouse ID (if None, use the default from config)
            
        Returns:
            Spark DataFrame with query results
        """
        try:
            warehouse = warehouse_id or self.config.get_warehouse_id()
            logger.info(f"Executing SQL query using warehouse: {warehouse}")
            
            # Use spark.sql for executing queries
            df = self.spark.sql(query)
            logger.info(f"Query executed successfully, returned {df.count()} rows")
            
            return df
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            raise
    
    def load_unstructured_data(self, file_paths: Union[str, List[str]], file_format: str = "text") -> List[Dict[str, Any]]:
        """
        Load unstructured data from files
        
        Args:
            file_paths: Path or list of paths to files
            file_format: Format of files (text, pdf, etc.)
            
        Returns:
            List of documents with content and metadata
        """
        try:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
                
            logger.info(f"Loading {len(file_paths)} unstructured {file_format} files")
            
            documents = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                if file_format.lower() == "text":
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    # Create document with metadata
                    document = {
                        "id": os.path.basename(file_path),
                        "title": os.path.basename(file_path),
                        "content": content,
                        "metadata": {
                            "source": file_path,
                            "format": file_format,
                            "size_bytes": os.path.getsize(file_path),
                            "created_at": pd.Timestamp.now().isoformat()
                        }
                    }
                    documents.append(document)
                else:
                    logger.warning(f"Unsupported file format: {file_format}")
            
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading unstructured data: {str(e)}")
            raise
