"""
Module for Mosaic AI endpoint creation and management.
"""

import time
import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

logger = logging.getLogger(__name__)

class MosaicEndpoint:
    """Class for managing Mosaic AI endpoints"""
    
    def __init__(self, config, workspace_client=None):
        """
        Initialize Mosaic endpoint manager
        
        Args:
            config: Configuration object
            workspace_client: Optional WorkspaceClient
        """
        self.config = config
        self.workspace_client = workspace_client or self.initialize_workspace_client()
    
    def initialize_workspace_client(self):
        """
        Initialize workspace client with service principal if in production
        
        Returns:
            Initialized WorkspaceClient
        """
        try:
            # In production, use service principal
            if self.config.environment == "prod":
                logger.info("Initializing workspace client with service principal")
                client = WorkspaceClient(
                    host=self.config.base_config.get("databricks_host", "https://databricks.com"),
                    client_id=self.config.base_config["service_principal"]["client_id"],
                    client_secret=self.config.base_config["service_principal"]["client_secret"]
                )
            else:
                # In dev/qa, use current authentication
                logger.info(f"Initializing workspace client with default authentication for {self.config.environment}")
                client = WorkspaceClient()
            
            return client
        except Exception as e:
            logger.error(f"Failed to initialize workspace client: {str(e)}")
            raise
    
    def create_mosaic_ai_endpoint(self):
        """
        Create or update Mosaic AI endpoint
        
        Returns:
            Endpoint name
        """
        try:
            # Select endpoint name based on environment
            endpoint_name = self.config.get_endpoint_name()
            
            # Check if endpoint already exists
            try:
                endpoint = self.workspace_client.serving_endpoints.get(endpoint_name)
                logger.info(f"Endpoint {endpoint_name} already exists, updating configuration")
                endpoint_exists = True
            except:
                logger.info(f"Creating new endpoint {endpoint_name}")
                endpoint_exists = False
            
            # Define endpoint configuration based on environment
            if self.config.environment == "prod":
                # Production environment with provisioned throughput
                workload_size = "Small"
                workload_type = "GPU_SMALL"
                scale_to_zero_enabled = False
                provisioned_throughput = self.config.active_config["provisioned_throughput"]
            else:
                # Dev/QA environments with scale-to-zero
                workload_size = "Small"
                workload_type = "GPU_SMALL"
                scale_to_zero_enabled = self.config.active_config["scale_to_zero"]
                provisioned_throughput = None
            
            # Configure the model serving endpoint
            served_models = [
                ServedModelInput(
                    model_name=self.config.base_config["uc_config"]["model_name"],
                    model_version=self.config.base_config["model"]["model_version"],
                    workload_size=workload_size,
                    workload_type=workload_type,
                    scale_to_zero_enabled=scale_to_zero_enabled
                )
            ]
            
            if provisioned_throughput:
                endpoint_config = EndpointCoreConfigInput(
                    name=endpoint_name,
                    served_models=served_models,
                    traffic_config={"routes": [{"served_model_name": self.config.base_config["uc_config"]["model_name"], "traffic_percentage": 100}]},
                    provisioned_throughput=provisioned_throughput
                )
            else:
                endpoint_config = EndpointCoreConfigInput(
                    name=endpoint_name,
                    served_models=served_models,
                    traffic_config={"routes": [{"served_model_name": self.config.base_config["uc_config"]["model_name"], "traffic_percentage": 100}]}
                )
            
            # Create or update the endpoint
            if endpoint_exists:
                self.workspace_client.serving_endpoints.update_config(
                    name=endpoint_name,
                    config=endpoint_config
                )
            else:
                self.workspace_client.serving_endpoints.create(
                    name=endpoint_name,
                    config=endpoint_config
                )
            
            # Wait for endpoint to be ready
            max_wait = 300  # 5 minutes
            wait_time = 0
            while wait_time < max_wait:
                status = self.workspace_client.serving_endpoints.get(endpoint_name).state.ready
                if status:
                    break
                time.sleep(15)
                wait_time += 15
                logger.info(f"Waiting for endpoint {endpoint_name} to be ready... ({wait_time}s)")
            
            if wait_time >= max_wait:
                logger.warning(f"Endpoint {endpoint_name} creation timed out")
            else:
                logger.info(f"Endpoint {endpoint_name} is ready")
            
            return endpoint_name
        
        except Exception as e:
            logger.error(f"Failed to create Mosaic AI endpoint: {str(e)}")
            raise
