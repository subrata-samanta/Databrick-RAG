"""
Main entry point for the RAG pipeline application.
"""

import os
import logging
from pyspark.sql import SparkSession

# Import modules from our application
from config.config import Config
from data.data_generator import DataGenerator
from data.data_loader import DataLoader
from data.data_storage import DataStorage
from models.embeddings import EmbeddingManager
from models.llm import LLMManager
from models.model_registry import ModelRegistry
from endpoints.mosaic_endpoint import MosaicEndpoint
from endpoints.ai_gateway import AIGateway
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGChainManager
from rag.sql_agent import SQLAgentManager
from agents.unified_agent import UnifiedAgent
from monitoring.lakehouse_monitor import LakehouseMonitoring
from monitoring.dashboard import DashboardManager
from monitoring.inference_logger import InferenceLogger
from jobs.etl_job import ETLJobManager
from utils.error_handling import ErrorHandler
from utils.optimization import OptimizationManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main class for orchestrating the RAG pipeline"""
    
    def __init__(self, environment="dev"):
        """
        Initialize the RAG pipeline
        
        Args:
            environment: Deployment environment (dev, qa, prod)
        """
        # Load configuration
        self.config = Config(environment)
        logger.info(f"Initializing RAG pipeline in {environment} environment")
        
        # Initialize Spark session
        self.spark = SparkSession.builder.getOrCreate()
        
        # Initialize components
        self.data_generator = DataGenerator()
        self.data_loader = DataLoader(self.spark, self.config)
        self.data_storage = DataStorage(self.spark, self.config)
        self.embedding_manager = EmbeddingManager(self.spark, self.config)
        self.llm_manager = LLMManager(self.config)
        self.model_registry = ModelRegistry(self.config)
        self.mosaic_endpoint = MosaicEndpoint(self.config)
        self.ai_gateway = AIGateway(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)
        self.monitoring = LakehouseMonitoring(self.config)
        self.dashboard_manager = DashboardManager(self.config)
        self.etl_job_manager = ETLJobManager(self.config)
        self.optimization_manager = OptimizationManager(self.config)
        
        # These will be initialized later in the pipeline
        self.rag_chain = None
        self.sql_agent = None
        self.unified_agent = None
        self.error_handler = None
        self.inference_table = None
        self.inference_logger = None
    
    def generate_data(self):
        """Generate synthetic data and save to storage"""
        logger.info("Generating synthetic data")
        
        # Generate documents
        documents = self.data_generator.generate_documents(num_docs=200)
        documents_df = self.spark.createDataFrame(documents)
        
        # Generate structured data
        structured_data = self.data_generator.generate_structured_data()
        customers_df = self.spark.createDataFrame(structured_data["customers"])
        products_df = self.spark.createDataFrame(structured_data["products"])
        orders_df = self.spark.createDataFrame(structured_data["orders"])
        
        # Save data
        self.documents_table = self.data_storage.save_documents_to_volume(documents_df)
        self.data_storage.save_structured_data_to_tables(customers_df, products_df, orders_df)
        
        logger.info("Synthetic data generated and saved")
    
    def setup_vector_search(self):
        """Set up vector search for documents"""
        logger.info("Setting up vector search")
        
        # Create embeddings and vector index
        self.embeddings_table, self.vector_index = self.embedding_manager.create_embeddings_and_vector_index()
        
        logger.info(f"Vector search set up with index: {self.vector_index}")
    
    def setup_model(self):
        """Register model with MLflow"""
        logger.info("Setting up MLflow model")
        
        # Register model
        self.model_name, self.model_version, self.run_id = self.model_registry.register_model_with_mlflow()
        
        logger.info(f"Model registered: {self.model_name} version {self.model_version}")
    
    def setup_endpoint(self):
        """Set up Mosaic AI endpoint"""
        logger.info("Setting up Mosaic AI endpoint")
        
        # Create endpoint
        self.endpoint_name = self.mosaic_endpoint.create_mosaic_ai_endpoint()
        
        # Set up inference tables
        self.inference_table = self.data_storage.setup_inference_tables()
        
        # Initialize inference logger
        self.inference_logger = InferenceLogger(self.spark, self.config)
        
        # Set up AI Gateway
        self.gateway_name = self.ai_gateway.setup_ai_gateway(self.inference_table)
        
        logger.info(f"Mosaic AI endpoint set up: {self.endpoint_name}")
    
    def setup_agents(self):
        """Set up RAG chain, SQL agent, and unified agent"""
        logger.info("Setting up agents")
        
        # Initialize LLM
        llm = self.llm_manager.initialize_llm()
        
        # Set up vector store retriever
        self.retriever = self.vector_store_manager.setup_vector_store()
        
        # Create RAG chain
        rag_chain_manager = RAGChainManager(self.config, self.retriever)
        self.rag_chain = rag_chain_manager.create_rag_chain()
        
        # Create SQL agent
        sql_agent_manager = SQLAgentManager(self.config)
        self.sql_agent = sql_agent_manager.create_sql_agent()
        
        # Create unified agent
        self.unified_agent = UnifiedAgent(self.config, self.rag_chain, self.sql_agent, self.spark)
        self.agent = self.unified_agent.create_unified_agent()
        
        # Set inference table for logging
        self.unified_agent.set_inference_table(self.inference_table)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.unified_agent)
        
        logger.info("Agents set up successfully")
    
    def setup_monitoring(self):
        """Set up monitoring and dashboards"""
        logger.info("Setting up monitoring")
        
        # Set up Lakehouse monitoring
        self.metrics_table = self.monitoring.setup_lakehouse_monitoring(self.inference_table)
        
        # Create dashboard
        self.dashboard_config = self.dashboard_manager.create_monitoring_dashboard(self.inference_table)
        
        logger.info(f"Monitoring set up with metrics table: {self.metrics_table}")
    
    def setup_jobs(self):
        """Set up ETL jobs"""
        logger.info("Setting up ETL jobs")
        
        # Create ETL job
        self.etl_job_id = self.etl_job_manager.create_etl_job()
        
        logger.info(f"ETL job created with ID: {self.etl_job_id}")
    
    def apply_optimizations(self):
        """Apply cost and latency optimizations"""
        logger.info("Applying optimizations")
        
        # Apply optimizations
        self.optimization_config = self.optimization_manager.implement_optimizations()
        
        # Set up caching
        self.caching_config = self.optimization_manager.implement_caching_strategy()
        
        logger.info(f"Optimizations applied for {self.config.environment} environment")
    
    def run(self):
        """Run the complete pipeline setup"""
        try:
            logger.info("Starting RAG pipeline setup")
            
            # Generate synthetic data
            self.generate_data()
            
            # Set up vector search
            self.setup_vector_search()
            
            # Set up model and registry
            self.setup_model()
            
            # Set up endpoint
            self.setup_endpoint()
            
            # Apply optimizations
            self.apply_optimizations()
            
            # Set up agents
            self.setup_agents()
            
            # Set up monitoring
            self.setup_monitoring()
            
            # Set up ETL jobs
            self.setup_jobs()
            
            logger.info("RAG pipeline setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up RAG pipeline: {str(e)}")
            return False
    
    def ask(self, query):
        """
        Ask a question to the agent
        
        Args:
            query: Question to ask
            
        Returns:
            Agent response
        """
        if not self.unified_agent or not self.agent:
            raise Exception("Pipeline not initialized. Call run() first.")
        
        # Use error handler for enhanced query processing
        return self.error_handler.enhanced_ask_agent(query)

if __name__ == "__main__":
    # Get environment from environment variable or use default
    environment = os.environ.get("RAG_ENVIRONMENT", "dev")
    
    # Initialize and run the pipeline
    pipeline = RAGPipeline(environment)
    success = pipeline.run()
    
    if success:
        # Test with sample query
        print("\nTesting RAG pipeline with sample query:")
        response = pipeline.ask("What information do we have about Machine Learning?")
        print(f"Query: {response['query']}")
        print(f"Response: {response['response']}")
        print(f"Latency: {response['latency_ms']} ms")
    else:
        print("Failed to set up RAG pipeline")
