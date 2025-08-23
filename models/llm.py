"""
Module for LLM model utilities and interactions.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from langchain_databricks.chat_models import ChatDatabricks
from langchain.callbacks import get_openai_callback
from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)

class LLMManager:
    """Class for managing LLM interactions and utilities"""
    
    def __init__(self, config):
        """
        Initialize LLM manager
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.llm = None
        self.models = {
            "default": config.get_llm_model(),
            "embedding": config.get_embedding_model(),
        }
    
    def initialize_llm(self, model_name: Optional[str] = None, **kwargs) -> ChatDatabricks:
        """
        Initialize LLM with specific parameters
        
        Args:
            model_name: Optional model name to use (defaults to config value)
            kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Initialized LLM
        """
        try:
            model = model_name or self.models["default"]
            
            # Get default parameters based on environment
            if hasattr(self.config, 'environment'):
                if self.config.environment == "prod":
                    default_params = {
                        "temperature": 0.3,
                        "max_tokens": 4096,
                        "top_p": 0.9
                    }
                elif self.config.environment == "qa":
                    default_params = {
                        "temperature": 0.5,
                        "max_tokens": 2048,
                        "top_p": 0.9
                    }
                else:  # dev
                    default_params = {
                        "temperature": 0.7,
                        "max_tokens": 1024,
                        "top_p": 0.9
                    }
            else:
                default_params = {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.9
                }
            
            # Override defaults with any provided parameters
            params = {**default_params, **kwargs}
            
            # Initialize the LLM
            llm = ChatDatabricks(
                endpoint=model,
                **params
            )
            
            self.llm = llm
            logger.info(f"Initialized LLM with model {model}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def get_llm(self) -> ChatDatabricks:
        """
        Get the current LLM or initialize a new one
        
        Returns:
            LLM instance
        """
        if not self.llm:
            return self.initialize_llm()
        return self.llm
    
    def predict(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Make a prediction with the LLM with tracking
        
        Args:
            prompt: Input prompt
            kwargs: Additional parameters for the LLM
            
        Returns:
            Dictionary with prediction results and metrics
        """
        try:
            llm = self.get_llm()
            
            # Track token usage and latency
            start_time = time.time()
            with get_openai_callback() as cb:
                response = llm.predict(prompt, **kwargs)
                
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            result = {
                "response": response,
                "latency_ms": latency_ms,
                "tokens": {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens
                },
                "model": llm.endpoint
            }
            
            logger.info(f"LLM prediction completed in {latency_ms:.2f}ms using {cb.total_tokens} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error during LLM prediction: {str(e)}")
            raise
    
    def batch_predict(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Make batch predictions with the LLM
        
        Args:
            prompts: List of input prompts
            kwargs: Additional parameters for the LLM
            
        Returns:
            List of dictionaries with prediction results and metrics
        """
        results = []
        for prompt in prompts:
            try:
                result = self.predict(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    "response": str(e),
                    "latency_ms": -1,
                    "tokens": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    },
                    "model": self.llm.endpoint if self.llm else None,
                    "error": str(e)
                })
        
        return results
