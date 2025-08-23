"""
Module for inference logging and tracking.
"""

import uuid
import time
import logging
import pandas as pd
from typing import Dict, Any
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

class InferenceLogger:
    """Class for logging inference data"""
    
    def __init__(self, spark: SparkSession, config):
        """
        Initialize inference logger
        
        Args:
            spark: SparkSession
            config: Configuration object
        """
        self.spark = spark
        self.config = config
        self.inference_table = self.config.get_full_table_name("inference_logs")
        
        # Ensure the inference table exists
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the inference table exists"""
        try:
            # Check if table exists
            tables = self.spark.catalog.listTables(self.config.get_catalog_name(), self.config.get_schema_name())
            table_exists = any(table.name == "inference_logs" for table in tables)
            
            if not table_exists:
                logger.info(f"Creating inference table: {self.inference_table}")
                self.spark.sql(f"""
                CREATE TABLE IF NOT EXISTS {self.inference_table} (
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
                    metadata MAP<STRING, STRING>,
                    date_day STRING
                )
                USING delta
                PARTITIONED BY (date_day)
                """)
        except Exception as e:
            logger.error(f"Error ensuring inference table exists: {str(e)}")
    
    def log_inference(self, query: str, response: str, latency_ms: float, tokens_input: int = None, 
                     tokens_output: int = None, user_id: str = "system", metadata: Dict[str, str] = None):
        """
        Log inference data to the inference table
        
        Args:
            query: User query
            response: Model response
            latency_ms: Latency in milliseconds
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            user_id: User identifier
            metadata: Additional metadata
        """
        try:
            # Calculate token counts if not provided
            if tokens_input is None:
                tokens_input = len(query.split())
            if tokens_output is None:
                tokens_output = len(str(response).split())
            
            # Create inference log entry
            log_entry = {
                "request_id": str(uuid.uuid4()),
                "timestamp": pd.Timestamp.now(),
                "query": query,
                "response": str(response),
                "latency_ms": latency_ms,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "environment": self.config.environment,
                "endpoint_name": self.config.get_endpoint_name(),
                "model_name": self.config.get_llm_model(),
                "user_id": user_id,
                "metadata": metadata or {"source": "inference_logger"},
                "date_day": pd.Timestamp.now().strftime("%Y-%m-%d")
            }
            
            # Insert into inference logs table
            self.spark.createDataFrame([log_entry]).write.format("delta").mode("append").saveAsTable(self.inference_table)
            
            logger.info(f"Logged inference data to {self.inference_table}")
            
        except Exception as e:
            logger.error(f"Failed to log inference data: {str(e)}")
    
    def get_logs(self, start_time: str = None, end_time: str = None, 
                user_id: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve logs from the inference table
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            user_id: Optional user ID filter
            limit: Maximum number of logs to return
            
        Returns:
            Pandas DataFrame with inference logs
        """
        try:
            query = f"SELECT * FROM {self.inference_table}"
            conditions = []
            
            if start_time:
                conditions.append(f"timestamp >= '{start_time}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time}'")
            if user_id:
                conditions.append(f"user_id = '{user_id}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            df = self.spark.sql(query).toPandas()
            logger.info(f"Retrieved {len(df)} inference logs")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving inference logs: {str(e)}")
            raise
