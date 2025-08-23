"""
Module for ETL job creation and management.
"""

import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

logger = logging.getLogger(__name__)

class ETLJobManager:
    """Class for managing ETL jobs"""
    
    def __init__(self, config, workspace_client=None):
        """
        Initialize ETL job manager
        
        Args:
            config: Configuration object
            workspace_client: Optional WorkspaceClient
        """
        self.config = config
        self.workspace_client = workspace_client or WorkspaceClient()
    
    def create_etl_job(self):
        """
        Create ETL job for document processing
        
        Returns:
            Job ID
        """
        try:
            # Define the job configuration
            job_name = f"rag_document_processing_{self.config.environment}"
            notebook_path = "/Shared/rag_pipeline/document_processing"
            
            # Define notebook content first (in a real scenario, this would be in a separate file)
            notebook_content = """
# Document Processing ETL
import pyspark.sql.functions as F
from pyspark.sql.types import *
import json

# Get parameters
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
source_table = dbutils.widgets.get("source_table")

# Read documents
documents = spark.table(f"{catalog_name}.{schema_name}.{source_table}")

# Extract metadata fields for easier querying
processed_docs = documents.withColumn(
    "metadata_struct", 
    F.from_json(
        F.col("metadata"), 
        StructType([
            StructField("id", StringType(), True),
            StructField("title", StringType(), True),
            StructField("topics", ArrayType(StringType()), True),
            StructField("created_at", StringType(), True),
            StructField("source", StringType(), True),
            StructField("word_count", IntegerType(), True)
        ])
    )
).select(
    "id",
    "title",
    "content",
    F.col("metadata_struct.topics").alias("topics"),
    F.col("metadata_struct.created_at").alias("created_at"),
    F.col("metadata_struct.source").alias("source"),
    F.col("metadata_struct.word_count").alias("word_count"),
    "metadata"
)

# Write processed documents back to a new table
processed_docs.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.processed_documents"
)

print(f"Processed {processed_docs.count()} documents")
"""
            
            # Create the notebook
            try:
                self.workspace_client.workspace.mkdirs(path="/Shared/rag_pipeline")
            except:
                # Directory might already exist
                pass
            
            # Write notebook content
            self.workspace_client.workspace.import_notebook(
                content=notebook_content,
                path=notebook_path,
                language="PYTHON",
                overwrite=True
            )
            
            # Create the job
            job_config = jobs.JobSettings(
                name=job_name,
                tags={
                    "environment": self.config.environment,
                    "purpose": "rag_pipeline"
                },
                schedule=jobs.CronSchedule(
                    quartz_cron_expression="0 0 * * * ?",  # Daily at midnight
                    timezone_id="UTC"
                ),
                tasks=[
                    jobs.Task(
                        task_key="document_processing",
                        description="Process documents and extract metadata",
                        notebook_task=jobs.NotebookTask(
                            notebook_path=notebook_path,
                            base_parameters={
                                "catalog_name": self.config.get_catalog_name(),
                                "schema_name": self.config.get_schema_name(),
                                "source_table": self.config.base_config["vector_search"]["source_table_name"]
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
                email_notifications=jobs.JobEmailNotifications(
                    on_failure=["your-email@example.com"]
                )
            )
            
            # Create or update the job
            try:
                job_id = self.workspace_client.jobs.create(job_config)
                logger.info(f"Created ETL job with ID: {job_id}")
            except Exception as e:
                # Job might already exist, try to find and update it
                jobs_list = self.workspace_client.jobs.list(name=job_name)
                if jobs_list:
                    job_id = jobs_list[0].job_id
                    self.workspace_client.jobs.update(job_id=job_id, new_settings=job_config)
                    logger.info(f"Updated existing ETL job with ID: {job_id}")
                else:
                    raise e
                
            return job_id
        
        except Exception as e:
            logger.error(f"Failed to create ETL job: {str(e)}")
            raise
