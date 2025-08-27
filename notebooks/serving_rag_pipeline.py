# Databricks notebook source
# MAGIC %md
# MAGIC # Serving RAG Pipeline in Databricks
# MAGIC 
# MAGIC This notebook demonstrates how to serve the RAG pipeline in Databricks:
# MAGIC 
# MAGIC 1. Setting up the environment
# MAGIC 2. Deploying as an MLflow model
# MAGIC 3. Creating a Model Serving endpoint
# MAGIC 4. Testing the endpoint
# MAGIC 5. Integration examples

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Environment
# MAGIC 
# MAGIC First, let's install required dependencies and set up our project structure.

# COMMAND ----------

# MAGIC %pip install langchain langchain-databricks databricks-vectorsearch

# COMMAND ----------

# Set environment variables
import os
import sys

# Choose environment: "dev", "qa", or "prod"
environment = "dev"
os.environ["RAG_ENVIRONMENT"] = environment

# Add project root to path for imports
project_root = "/Workspace/Repos/rag_pipeline"  # Update this to your project path
if project_root not in sys.path:
    sys.path.append(project_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize and Test the RAG Pipeline
# MAGIC 
# MAGIC Let's import and test the RAG pipeline to ensure it works before deployment.

# COMMAND ----------

from main import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(environment)
success = pipeline.run()

if success:
    print("✅ RAG pipeline initialized successfully!")
    
    # Create a session
    session_id = pipeline.session_manager.create_session(user_id="notebook_user")
    
    # Test with a sample query
    response = pipeline.session_manager.ask(session_id, "What information do we have about Machine Learning?")
    print(f"\nQuery: {response.get('query', 'N/A')}")
    print(f"Response: {response.get('response', 'N/A')}")
    print(f"Latency: {response.get('latency_ms', 'N/A')} ms")
else:
    print("❌ Failed to initialize RAG pipeline")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Package and Deploy as an MLflow Model
# MAGIC 
# MAGIC Now we'll package the RAG pipeline as an MLflow model for deployment.

# COMMAND ----------

import pandas as pd
import mlflow
from mlflow.models import infer_signature

# Set MLflow registry URI to Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

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

# COMMAND ----------

# Create example input for signature
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

# COMMAND ----------

# Start an MLflow run for tracking
with mlflow.start_run(run_name=f"rag_deployment_{environment}") as run:
    # Log environment parameters
    mlflow.log_params({
        "environment": environment,
        "embedding_model": pipeline.config.get_embedding_model(),
        "llm_model": pipeline.config.get_llm_model()
    })
    
    # Log the model
    model_info = mlflow.pyfunc.log_model(
        python_model=RagPipelineWrapper(pipeline),
        artifact_path="rag_pipeline_model",
        registered_model_name=f"{pipeline.config.get_catalog_name()}.{pipeline.config.get_schema_name()}.rag_pipeline_serving",
        signature=signature,
        input_example=example_input
    )
    
    print(f"RAG pipeline packaged as MLflow model: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Deploy to a Model Serving Endpoint
# MAGIC 
# MAGIC Now we'll deploy the model to a Databricks Model Serving endpoint.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving

# Initialize workspace client
workspace_client = WorkspaceClient()

# COMMAND ----------

# Get the latest version of the registered model
client = mlflow.MlflowClient()
model_name = f"{pipeline.config.get_catalog_name()}.{pipeline.config.get_schema_name()}.rag_pipeline_serving"
latest_versions = client.get_latest_versions(model_name)
latest_version = latest_versions[0].version

print(f"Latest model version: {latest_version}")

# COMMAND ----------

# Define endpoint name based on environment
endpoint_name = f"rag-pipeline-{environment}"

# Check if endpoint already exists
try:
    endpoint = workspace_client.serving_endpoints.get(endpoint_name)
    print(f"Endpoint {endpoint_name} already exists, updating config")
    endpoint_exists = True
except:
    print(f"Creating new endpoint {endpoint_name}")
    endpoint_exists = False

# COMMAND ----------

# Configure workload size and scaling based on environment
if environment == "prod":
    # Production configuration with provisioned throughput
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
    workspace_client.serving_endpoints.update_config(
        name=endpoint_name,
        config=config
    )
else:
    workspace_client.serving_endpoints.create(
        name=endpoint_name,
        config=config
    )

print(f"Serving endpoint {endpoint_name} deployed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test the Serving Endpoint
# MAGIC 
# MAGIC Let's test our deployed endpoint.

# COMMAND ----------

import requests
import json

# Get workspace URL
workspace_url = workspace_client.config.host

# For notebook testing, we'll use databricks.model_serving.get_deployment_url API
# which works within notebooks
def query_endpoint(endpoint_name, data):
    import json
    import os
    from databricks.model_serving import get_deployment_url

    # Get the url of the endpoint and attach the invocations path
    url = get_deployment_url(endpoint_name) + "/invocations" 

    # Set the headers and data
    headers = {"Content-Type": "application/json"}
    data_json = json.dumps(data)
    
    # Make the request
    response = requests.post(url, headers=headers, data=data_json)
    return response

# COMMAND ----------

# Test query
test_data = {"dataframe_records": [
    {"query": "What information do we have about Machine Learning?", "session_id": "notebook-test-session"}
]}

response = query_endpoint(endpoint_name, test_data)
print(f"Status code: {response.status_code}")
print("Response:")
print(json.dumps(response.json(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. API Usage Examples
# MAGIC 
# MAGIC Here are examples for calling the endpoint from different clients.

# COMMAND ----------

# Generate API examples
workspace_url = workspace_client.config.host

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

print("## Python Example")
print("```python")
print(python_example)
print("```")

# COMMAND ----------

# curl example
curl_example = f"""
curl -X POST {workspace_url}/serving-endpoints/{endpoint_name}/invocations \\
  -H "Authorization: Bearer YOUR_PERSONAL_ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"dataframe_records": [{{"query": "What information do we have about Machine Learning?", "session_id": "session-123"}}]}}'
"""

print("## cURL Example")
print("```bash")
print(curl_example)
print("```")

# COMMAND ----------

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

print("## JavaScript Example")
print("```javascript")
print(js_example)
print("```")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Creating a Scheduled Job for Deployment
# MAGIC 
# MAGIC Let's create a job that can be scheduled to deploy the RAG pipeline.

# COMMAND ----------

from databricks.sdk.service import jobs

# Create job configuration
job_config = jobs.JobSettings(
    name=f"Deploy RAG Pipeline - {environment}",
    tags={
        "environment": environment,
        "purpose": "rag_deployment"
    },
    tasks=[
        jobs.Task(
            task_key="deploy_rag_pipeline",
            description="Deploy RAG pipeline to a model serving endpoint",
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{project_root}/notebooks/serving_rag_pipeline",
                base_parameters={
                    "environment": environment
                }
            ),
            new_cluster=jobs.ClusterSpec(
                spark_version="13.3.x-scala2.12",
                node_type_id="Standard_DS3_v2",
                num_workers=1,
                spark_conf={
                    "spark.databricks.cluster.profile": "singleNode",
                    "spark.master": "local[*]"
                }
            )
        )
    ],
    schedule=jobs.CronSchedule(
        quartz_cron_expression="0 0 * * 0",  # Weekly on Sunday at midnight
        timezone_id="UTC"
    ),
    email_notifications=jobs.JobEmailNotifications(
        on_failure=["admin@example.com"]
    )
)

# Create the job
try:
    job_id = workspace_client.jobs.create(job_config)
    print(f"Created job with ID: {job_id}")
except Exception as e:
    print(f"Error creating job: {str(e)}")
