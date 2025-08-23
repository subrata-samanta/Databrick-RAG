"""
Module for error handling and exception management.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Custom exceptions for the RAG pipeline
class RAGException(Exception):
    """Base exception class for RAG pipeline errors."""
    pass

class ModelInferenceError(RAGException):
    """Exception raised for errors during model inference."""
    pass

class RetrievalError(RAGException):
    """Exception raised for errors during document retrieval."""
    pass

class DatabaseError(RAGException):
    """Exception raised for database-related errors."""
    pass

class ConfigurationError(RAGException):
    """Exception raised for configuration-related errors."""
    pass

class ErrorHandler:
    """Class for handling errors in the RAG pipeline"""
    
    def __init__(self, unified_agent=None, max_retries=2):
        """
        Initialize error handler
        
        Args:
            unified_agent: Optional UnifiedAgent for logging
            max_retries: Maximum number of retries for failed operations
        """
        self.unified_agent = unified_agent
        self.max_retries = max_retries
    
    def with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic
        
        Args:
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                last_error = e
                
                # Log the error
                logger.error(f"Error (retry {retry_count}/{self.max_retries}): {str(e)}")
                
                # If we still have retries left, wait and try again
                if retry_count <= self.max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # Log final failure
                    logger.error(f"Failed after {self.max_retries} retries: {str(e)}")
                    raise
    
    def enhanced_ask_agent(self, query: str) -> Dict[str, Any]:
        """
        Enhanced version of ask_agent with retry logic and detailed error handling
        
        Args:
            query: User query
            
        Returns:
            Response dictionary
        """
        if not self.unified_agent or not self.unified_agent.agent:
            raise ConfigurationError("Unified agent not properly initialized")
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Measure execution time
                start_time = time.time()
                
                # Add request ID for tracking
                request_id = str(uuid.uuid4())
                
                # Log the incoming query
                logger.info(f"Processing query (ID: {request_id}): {query}")
                
                # Process the query
                try:
                    # First try the RAG chain
                    if "document" in query.lower() or "article" in query.lower() or "content" in query.lower():
                        logger.info(f"Using RAG retrieval for query: {query}")
                        rag_response = self.unified_agent.rag_chain({"query": query})
                        response = rag_response["result"]
                        source_docs = rag_response["source_documents"]
                        
                        # Format response with sources
                        source_info = "\n\nSources:\n"
                        for i, doc in enumerate(source_docs):
                            doc_id = doc.metadata.get("id", f"doc-{i}")
                            source_info += f"- {doc_id}: {doc.metadata.get('title', 'Untitled')}\n"
                        
                        response = response + source_info
                        
                    # Try SQL agent for data-related queries
                    elif any(term in query.lower() for term in ["data", "report", "sales", "customer", "product", "order", "revenue"]):
                        logger.info(f"Using SQL agent for query: {query}")
                        response = self.unified_agent.sql_agent.run(query)
                        
                    # Use the unified agent as fallback
                    else:
                        logger.info(f"Using unified agent for query: {query}")
                        response = self.unified_agent.agent.run(query)
                
                except Exception as e:
                    logger.warning(f"Error in specific agent, falling back to unified agent: {str(e)}")
                    # Fallback to the unified agent
                    response = self.unified_agent.agent.run(query)
                
                # Calculate latency
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Log the inference
                if self.unified_agent.inference_table:
                    self.unified_agent.log_inference(query, response, latency_ms)
                
                # Return successful response
                logger.info(f"Query processed successfully (ID: {request_id}, latency: {latency_ms:.2f}ms)")
                return {
                    "request_id": request_id,
                    "query": query,
                    "response": response,
                    "latency_ms": round(latency_ms, 2),
                    "status": "success"
                }
                
            except Exception as e:
                retry_count += 1
                last_error = e
                
                # Log the error
                logger.error(f"Error processing query (retry {retry_count}/{self.max_retries}): {str(e)}")
                
                # Determine error type for specific handling
                if "model inference" in str(e).lower() or "endpoint" in str(e).lower():
                    error_type = "MODEL_INFERENCE"
                    specific_error = ModelInferenceError(f"Model inference error: {str(e)}")
                elif "retriev" in str(e).lower() or "vector" in str(e).lower() or "embedding" in str(e).lower():
                    error_type = "RETRIEVAL"
                    specific_error = RetrievalError(f"Retrieval error: {str(e)}")
                elif "sql" in str(e).lower() or "database" in str(e).lower() or "query" in str(e).lower():
                    error_type = "DATABASE"
                    specific_error = DatabaseError(f"Database error: {str(e)}")
                else:
                    error_type = "GENERAL"
                    specific_error = RAGException(f"General error: {str(e)}")
                
                # If we still have retries left, wait and try again
                if retry_count <= self.max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # Log final failure
                    logger.error(f"Failed to process query after {self.max_retries} retries: {query}")
                    
                    # Return error response
                    return {
                        "request_id": str(uuid.uuid4()),
                        "query": query,
                        "response": f"I'm sorry, I encountered an error ({error_type}). Please try again or contact support if the issue persists.",
                        "latency_ms": -1,
                        "status": "error",
                        "error_type": error_type,
                        "error_message": str(specific_error)
                    }
