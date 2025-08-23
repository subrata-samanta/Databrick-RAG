"""
Module for Lakehouse monitoring for GenAI.
"""

import logging
from databricks.lakehouse_monitoring import Monitor

logger = logging.getLogger(__name__)

class LakehouseMonitoring:
    """Class for setting up Lakehouse monitoring for GenAI"""
    
    def __init__(self, config):
        """
        Initialize Lakehouse monitoring
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def setup_lakehouse_monitoring(self, inference_table):
        """
        Set up Lakehouse monitoring for inference logs
        
        Args:
            inference_table: Table name for inference logs
            
        Returns:
            Monitoring metrics table name
        """
        try:
            catalog_name = self.config.get_catalog_name()
            schema_name = self.config.get_schema_name()
            
            # Set up monitoring on the inference logs table
            monitor = Monitor(
                table_name=inference_table,
                time_column="timestamp"
            )
            
            # Define metrics to monitor
            monitor.add_timestamp_metric()
            monitor.add_metric(
                name="avg_latency",
                function="AVG",
                columns=["latency_ms"]
            )
            monitor.add_metric(
                name="p95_latency",
                function="PERCENTILE",
                columns=["latency_ms"],
                params={"percentile": 0.95}
            )
            monitor.add_metric(
                name="request_count",
                function="COUNT",
                columns=["request_id"]
            )
            monitor.add_metric(
                name="avg_tokens_output",
                function="AVG",
                columns=["tokens_output"]
            )
            
            # Add dimensions for slicing
            monitor.add_dimension(["environment", "endpoint_name", "model_name", "date_day"])
            
            # Start monitoring
            monitoring_table = f"{catalog_name}.{schema_name}.inference_metrics"
            monitor.start_monitoring(output_table_name=monitoring_table)
            
            logger.info(f"Started Lakehouse monitoring for inference logs, metrics stored in {monitoring_table}")
            return monitoring_table
            
        except Exception as e:
            logger.error(f"Failed to setup Lakehouse monitoring: {str(e)}")
            raise
