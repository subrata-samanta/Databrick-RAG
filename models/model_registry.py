"""
Module for MLflow model registration with Unity Catalog.
"""

import json
import shutil
import tempfile
import time
import logging
import mlflow
from mlflow.pyfunc import PythonModel

logger = logging.getLogger(__name__)

class RAGAssistant(mlflow.pyfunc.PythonModel):
    """RAG Assistant model for MLflow registration"""
    
    def __init__(self, config):
        self.config = config
    
    def load_context(self, context):
        """Load model context when loading from MLflow"""
        self.config = context.artifacts["config"]
    
    def predict(self, context, model_input):
        """Prediction method (placeholder)"""
        # This is a placeholder implementation
        # In production, you would include actual inference logic
        return {
            "status": "Model loaded successfully",
            "config": self.config,
            "input_shape": model_input.shape
        }

class ModelRegistry:
    """Class for managing model registration with MLflow"""
    
    def __init__(self, config):
        """
        Initialize model registry manager
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def register_model_with_mlflow(self):
        """
        Register model with MLflow and Unity Catalog
        
        Returns:
            Tuple of (model_name, model_version, run_id)
        """
        try:
            # Set MLflow registry URI to point to UC
            mlflow.set_registry_uri('databricks-uc')
            
            catalog_name = self.config.get_catalog_name()
            schema_name = self.config.get_schema_name()
            model_name = self.config.base_config["uc_config"]["model_name"]
            
            # Fully qualified model name
            registered_model_name = f"{catalog_name}.{schema_name}.{model_name}"
            
            # Start MLflow run for model registration
            with mlflow.start_run(run_name="rag_assistant_registration") as run:
                # Log model configuration as parameters
                mlflow.log_params({
                    "embedding_model": self.config.get_embedding_model(),
                    "llm_model": self.config.get_llm_model(),
                    "vector_index": self.config.base_config["uc_config"]["vector_index_name"],
                    "environment": self.config.environment
                })
                
                # Create artifacts directory
                artifacts_dir = tempfile.mkdtemp()
                try:
                    # Save config as artifact
                    with open(f"{artifacts_dir}/config.json", "w") as f:
                        json.dump(self.config.base_config, f)
                    
                    # Log the model with its config
                    mlflow.pyfunc.log_model(
                        artifact_path="rag_assistant",
                        python_model=RAGAssistant(self.config.base_config),
                        artifacts={"config": f"{artifacts_dir}/config.json"},
                        registered_model_name=registered_model_name
                    )
                finally:
                    shutil.rmtree(artifacts_dir)
                
                # Get run ID for reference
                run_id = run.info.run_id
            
            # Wait for the model to be registered
            time.sleep(5)
            
            # Get model version
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            latest_version = versions[0].version
            
            logger.info(f"Registered model {registered_model_name} version {latest_version} with MLflow")
            return registered_model_name, latest_version, run_id
        
        except Exception as e:
            logger.error(f"Failed to register model with MLflow: {str(e)}")
            raise
