"""
Module for unified agent framework.
"""

import uuid
import time
import logging
import pandas as pd
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain_databricks.chat_models import ChatDatabricks
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

class UnifiedAgent:
    """Class for managing unified agent framework"""
    
    def __init__(self, config, rag_chain, sql_agent, spark=None):
        """
        Initialize unified agent
        
        Args:
            config: Configuration object
            rag_chain: RAG chain for document retrieval
            sql_agent: SQL agent for text-to-SQL queries
            spark: Optional SparkSession
        """
        self.config = config
        self.rag_chain = rag_chain
        self.sql_agent = sql_agent
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.agent = None
        self.inference_table = None
    
    def create_unified_agent(self):
        """
        Create unified agent combining RAG and SQL capabilities
        
        Returns:
            Agent executor
        """
        try:
            # Define tools for our agent
            tools = [
                Tool(
                    name="RAG-Retrieval",
                    func=lambda query: self.rag_chain({"query": query})["result"],
                    description="Useful for answering questions about documents and knowledge base. Input should be a question about the content of documents."
                ),
                Tool(
                    name="SQL-Query",
                    func=lambda query: self.sql_agent.run(query),
                    description="Useful for querying structured data and generating reports. Input should be a question about business data that requires SQL queries."
                )
            ]
            
            # Define the prompt for the agent
            prefix = """You are an AI assistant with access to both document knowledge and structured data.
            You have access to the following tools:"""
            
            suffix = """Begin!

            Question: {input}
            {agent_scratchpad}"""
            
            # Create the agent
            prompt = ZeroShotAgent.create_prompt(
                tools, 
                prefix=prefix, 
                suffix=suffix,
                input_variables=["input", "agent_scratchpad"]
            )
            
            # Initialize LLM
            llm = ChatDatabricks(endpoint=self.config.get_llm_model())
            
            # Create memory for conversation history
            memory = ConversationBufferMemory(memory_key="chat_history")
            
            # Create the agent executor
            llm_chain = ZeroShotAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=llm_chain,
                tools=tools,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
            
            logger.info("Created unified agent with RAG and SQL capabilities")
            self.agent = agent_executor
            return agent_executor
            
        except Exception as e:
            logger.error(f"Failed to create unified agent: {str(e)}")
            raise
    
    def set_inference_table(self, inference_table):
        """
        Set the inference table for logging
        
        Args:
            inference_table: Table name for inference logging
        """
        self.inference_table = inference_table
    
    def log_inference(self, query, response, latency_ms):
        """
        Log inference data to table
        
        Args:
            query: User query
            response: Model response
            latency_ms: Latency in milliseconds
        """
        try:
            if not self.inference_table:
                logger.warning("Inference table not set, skipping logging")
                return
            
            # Create inference log entry
            log_entry = {
                "request_id": str(uuid.uuid4()),
                "timestamp": pd.Timestamp.now(),
                "query": query,
                "response": str(response),
                "latency_ms": latency_ms,
                "tokens_input": len(query.split()),  # Approximate
                "tokens_output": len(str(response).split()),  # Approximate
                "environment": self.config.environment,
                "endpoint_name": self.config.get_endpoint_name(),
                "model_name": self.config.get_llm_model(),
                "user_id": "demo_user",
                "metadata": {"source": "notebook_demo"},
                "date_day": pd.Timestamp.now().strftime("%Y-%m-%d")
            }
            
            # Insert into inference logs table
            self.spark.createDataFrame([log_entry]).write.format("delta").mode("append").saveAsTable(self.inference_table)
            
            logger.info(f"Logged inference data to {self.inference_table}")
            
        except Exception as e:
            logger.error(f"Failed to log inference data: {str(e)}")
    
    def ask(self, query):
        """
        Ask a question to the agent with error handling and logging
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query, response, and latency information
        """
        try:
            start_time = time.time()
            response = self.agent.run(query)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Log the inference
            self.log_inference(query, response, latency_ms)
            
            return {
                "query": query,
                "response": response,
                "latency_ms": round(latency_ms, 2)
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "latency_ms": -1
            }
