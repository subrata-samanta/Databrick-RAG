"""
Deployment script for serving the RAG pipeline in Databricks.
This file handles:
1. Packaging and deploying the RAG pipeline
2. Setting up a REST API endpoint
3. Configuring model serving
4. Integration with Databricks features
"""

import os
import mlflow
import logging
import json
from typing import Dict, Any
from mlflow.models import infer_signature
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving

# Import the RAG pipeline
from main import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGDeployment:
    """Class for deploying and serving the RAG pipeline in Databricks"""
    
    def __init__(self, environment="dev"):
        """
        Initialize the deployment
        
        Args:
            environment: Deployment environment (dev, qa, prod)
        """
        self.environment = environment
        self.workspace_client = WorkspaceClient()
        
        # Initialize the RAG pipeline
        self.pipeline = RAGPipeline(environment)
        
        # Set paths for artifacts
        self.artifacts_path = f"/dbfs/rag_pipeline/{environment}/artifacts"
        os.makedirs(self.artifacts_path, exist_ok=True)
    
    def package_pipeline(self):
        """
        Package the RAG pipeline for deployment
        
        Returns:
            Path to packaged model
        """
        logger.info("Packaging RAG pipeline for deployment")
        
        # Set MLflow tracking URI to current workspace
        mlflow.set_registry_uri('databricks-uc')
        
        # Start an MLflow run for tracking
        with mlflow.start_run(run_name=f"rag_deployment_{self.environment}") as run:
            # Log environment parameters
            mlflow.log_params({
                "environment": self.environment,
                "embedding_model": self.pipeline.config.get_embedding_model(),
                "llm_model": self.pipeline.config.get_llm_model()
            })
            
            # Create a Python model wrapper for the RAG pipeline
            class RagPipelineWrapper(mlflow.pyfunc.PythonModel):
                def __init__(self, pipeline):
                    self.pipeline = pipeline
                    
                def load_context(self, context):
                    # Additional loading code if needed
                    pass
                    
                def predict(self, context, model_input):
                    """
                    Process queries through the RAG pipeline
                    
                    Args:
                        model_input: DataFrame with columns:
                                    - query: text query
                                    - session_id: optional session ID
                    
                    Returns:
                        Dictionary with responses
                    """
                    results = []
                    
                    for _, row in model_input.iterrows():
                        query = row.get('query', '')
                        session_id = row.get('session_id', None)
                        
                        # Process through the RAG pipeline
                        response = self.pipeline.ask(query, session_id)
                        results.append(response)
                    
                    return results
            
            # Create example input for signature
            import pandas as pd
            example_input = pd.DataFrame([
                {"query": "What is machine learning?", "session_id": "example-session-1"},
                {"query": "How many customers do we have?", "session_id": "example-session-2"}
            ])
            
            example_output = [
                {"response": "Machine learning is...", "status": "success"},
                {"response": "We have X customers...", "status": "success"}
            ]
            
            # Infer signature
            signature = infer_signature(example_input, example_output)
            
            # Log the model
            model_info = mlflow.pyfunc.log_model(
                python_model=RagPipelineWrapper(self.pipeline),
                artifact_path="rag_pipeline_model",
                registered_model_name=f"{self.pipeline.config.get_catalog_name()}.{self.pipeline.config.get_schema_name()}.rag_pipeline_serving",
                signature=signature,
                input_example=example_input
            )
            
            logger.info(f"RAG pipeline packaged as MLflow model: {model_info.model_uri}")
            return model_info.model_uri
    
    def deploy_serving_endpoint(self, model_uri):
        """
        Deploy the packaged model to a serving endpoint
        
        Args:
            model_uri: URI of the packaged model
            
        Returns:
            Name of the created endpoint
        """
        logger.info(f"Deploying model {model_uri} to serving endpoint")
        
        # Get the latest version of the registered model
        client = mlflow.MlflowClient()
        model_name = f"{self.pipeline.config.get_catalog_name()}.{self.pipeline.config.get_schema_name()}.rag_pipeline_serving"
        latest_versions = client.get_latest_versions(model_name)
        latest_version = latest_versions[0].version
        
        # Define endpoint name based on environment
        endpoint_name = f"rag-pipeline-{self.environment}"
        
        # Check if endpoint already exists
        try:
            endpoint = self.workspace_client.serving_endpoints.get(endpoint_name)
            logger.info(f"Endpoint {endpoint_name} already exists, updating config")
            endpoint_exists = True
        except:
            logger.info(f"Creating new endpoint {endpoint_name}")
            endpoint_exists = False
        
        # Configure workload size and scaling based on environment
        if self.environment == "prod":
            # Production configuration
            config = serving.EndpointCoreConfigInput(
                name=endpoint_name,
                served_models=[
                    serving.ServedModelInput(
                        model_name=model_name,
                        model_version=latest_version,
                        workload_size="Small",
                        scale_to_zero_enabled=False,
                        workload_type="CPU",
                    )
                ],
                traffic_config={"routes": [{"served_model_name": model_name, "traffic_percentage": 100}]},
                auto_capture_config={"enabled": True}
            )
        else:
            # Dev/QA configuration with scale-to-zero
            config = serving.EndpointCoreConfigInput(
                name=endpoint_name,
                served_models=[
                    serving.ServedModelInput(
                        model_name=model_name,
                        model_version=latest_version,
                        workload_size="Small",
                        scale_to_zero_enabled=True,
                        workload_type="CPU",
                    )
                ],
                traffic_config={"routes": [{"served_model_name": model_name, "traffic_percentage": 100}]},
                auto_capture_config={"enabled": True}
            )
        
        # Create or update endpoint
        if endpoint_exists:
            self.workspace_client.serving_endpoints.update_config(
                name=endpoint_name,
                config=config
            )
        else:
            self.workspace_client.serving_endpoints.create(
                name=endpoint_name,
                config=config
            )
        
        logger.info(f"Serving endpoint {endpoint_name} deployed successfully")
        return endpoint_name
    
    def generate_api_examples(self, endpoint_name):
        """
        Generate example code snippets for calling the API
        
        Args:
            endpoint_name: Name of the serving endpoint
            
        Returns:
            Dictionary with code examples
        """
        workspace_url = self.workspace_client.config.host
        
        # Python example
        python_example = f"""
import requests
import pandas as pd
import json

# Replace with your Databricks personal access token
token = "YOUR_PERSONAL_ACCESS_TOKEN"

# Endpoint URL
url = "{workspace_url}/serving-endpoints/{endpoint_name}/invocations"

# Prepare the request payload
data = {{"dataframe_records": [
    {{"query": "What information do we have about Machine Learning?", "session_id": "session-123"}}
]}}

# Set headers
headers = {{
    "Authorization": f"Bearer {{token}}",
    "Content-Type": "application/json"
}}

# Make the request
response = requests.post(url, headers=headers, json=data)
print(response.json())
"""

        # curl example
        curl_example = f"""
curl -X POST {workspace_url}/serving-endpoints/{endpoint_name}/invocations \\
  -H "Authorization: Bearer YOUR_PERSONAL_ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"dataframe_records": [{{"query": "What information do we have about Machine Learning?", "session_id": "session-123"}}]}}'
"""

        # JavaScript example
        js_example = f"""
const fetch = require('node-fetch');

async function queryRAG() {{
  const response = await fetch('{workspace_url}/serving-endpoints/{endpoint_name}/invocations', {{
    method: 'POST',
    headers: {{
      'Authorization': 'Bearer YOUR_PERSONAL_ACCESS_TOKEN',
      'Content-Type': 'application/json'
    }},
    body: JSON.stringify({{
      dataframe_records: [
        {{ query: "What information do we have about Machine Learning?", session_id: "session-123" }}
      ]
    }})
  }});
  
  const data = await response.json();
  console.log(data);
}}

queryRAG();
"""

        return {
            "python": python_example,
            "curl": curl_example,
            "javascript": js_example
        }
    
    def deploy(self):
        """
        Run the complete deployment process
        
        Returns:
            Dictionary with deployment information
        """
        try:
            logger.info(f"Starting RAG pipeline deployment for {self.environment} environment")
            
            # First initialize the pipeline
            success = self.pipeline.run()
            if not success:
                raise Exception("Failed to initialize RAG pipeline")
            
            # Package the pipeline as an MLflow model
            model_uri = self.package_pipeline()
            
            # Deploy the model to a serving endpoint
            endpoint_name = self.deploy_serving_endpoint(model_uri)
            
            # Generate API examples
            api_examples = self.generate_api_examples(endpoint_name)
            
            # Write examples to a file for reference
            with open(f"{self.artifacts_path}/api_examples.md", "w") as f:
                f.write(f"# RAG Pipeline API Examples for {endpoint_name}\n\n")
                f.write("## Python Example\n")
                f.write("```python\n")
                f.write(api_examples["python"])
                f.write("\n```\n\n")
                f.write("## cURL Example\n")
                f.write("```bash\n")
                f.write(api_examples["curl"])
                f.write("\n```\n\n")
                f.write("## JavaScript Example\n")
                f.write("```javascript\n")
                f.write(api_examples["javascript"])
                f.write("\n```\n")
            
            logger.info(f"Deployment completed successfully. Endpoint: {endpoint_name}")
            
            return {
                "status": "success",
                "environment": self.environment,
                "model_uri": model_uri,
                "endpoint_name": endpoint_name,
                "api_examples_path": f"{self.artifacts_path}/api_examples.md"
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return {
                "status": "failed",
                "environment": self.environment,
                "error": str(e)
            }

if __name__ == "__main__":
    # Get environment from environment variable or use default
    environment = os.environ.get("RAG_ENVIRONMENT", "dev")
    
    # Run the deployment
    deployment = RAGDeployment(environment)
    result = deployment.deploy()
    
    print(json.dumps(result, indent=2))
