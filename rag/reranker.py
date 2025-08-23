"""
Module for reranking retrieved documents to improve relevance.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from langchain.schema import Document
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

class Reranker:
    """Class for reranking retrieved documents to improve relevance"""
    
    def __init__(self, config, spark=None, cross_encoder_endpoint=None):
        """
        Initialize reranker
        
        Args:
            config: Configuration object
            spark: Optional SparkSession
            cross_encoder_endpoint: Optional cross-encoder endpoint name
        """
        self.config = config
        self.spark = spark or SparkSession.builder.getOrCreate()
        
        # Use specified endpoint or default to MS-MARCO cross-encoder
        self.cross_encoder_endpoint = cross_encoder_endpoint or "databricks-ms-marco-cross-encoder"
        
        # Reranking parameters
        self.params = {
            "top_k": config.base_config.get("reranker", {}).get("top_k", 10),
            "threshold": config.base_config.get("reranker", {}).get("threshold", 0.5),
            "use_reciprocal_rank_fusion": config.base_config.get("reranker", {}).get("use_reciprocal_rank_fusion", True),
            "score_aggregation": config.base_config.get("reranker", {}).get("score_aggregation", "mean")
        }
        
        logger.info(f"Reranker initialized with endpoint: {self.cross_encoder_endpoint}")
    
    def compute_cross_encoder_scores(self, query: str, docs: List[Document]) -> List[float]:
        """
        Compute cross-encoder scores for query-document pairs
        
        Args:
            query: Query text
            docs: List of retrieved documents
            
        Returns:
            List of relevance scores
        """
        try:
            # Extract document contents
            doc_texts = [doc.page_content for doc in docs]
            
            # Create query-document pairs
            pairs = [[query, doc_text] for doc_text in doc_texts]
            
            # Use Databricks inference endpoint for cross-encoder scoring
            # Note: This is a simplified version, in production you would use the actual endpoint
            # We're simulating the behavior for demonstration
            
            # In a real implementation, you'd use something like:
            # scores = self.spark.sql(f"""
            #     SELECT DATABRICKS_INFERENCE('{self.cross_encoder_endpoint}', 
            #            NAMED_STRUCT('query', '{query}', 'documents', array({doc_texts_str})))
            #     AS scores
            # """).collect()[0][0]
            
            # Simulate cross-encoder scoring for now
            import random
            base_scores = np.array([0.5 + 0.5 * random.random() for _ in range(len(pairs))])
            
            logger.info(f"Computed cross-encoder scores for {len(docs)} documents")
            return base_scores.tolist()
            
        except Exception as e:
            logger.error(f"Error computing cross-encoder scores: {str(e)}")
            # Return default scores on error
            return [0.5] * len(docs)
    
    def rerank(self, query: str, docs: List[Document], 
               retriever_scores: Optional[List[float]] = None) -> List[Document]:
        """
        Rerank documents based on cross-encoder relevance
        
        Args:
            query: User query
            docs: List of retrieved documents
            retriever_scores: Optional scores from initial retriever
            
        Returns:
            Reranked list of documents
        """
        if not docs:
            return []
        
        # If only one document, return as is
        if len(docs) == 1:
            return docs
        
        try:
            # Compute cross-encoder scores
            cross_encoder_scores = self.compute_cross_encoder_scores(query, docs)
            
            # If retriever scores are provided, combine them with cross-encoder scores
            if retriever_scores and self.params["use_reciprocal_rank_fusion"]:
                # Reciprocal Rank Fusion
                retriever_ranks = np.argsort(np.argsort(-np.array(retriever_scores))) + 1
                ce_ranks = np.argsort(np.argsort(-np.array(cross_encoder_scores))) + 1
                
                # RRF formula: 1 / (k + rank)
                k = 60  # Typical RRF constant
                rrf_scores = 1.0 / (k + retriever_ranks) + 1.0 / (k + ce_ranks)
                final_scores = rrf_scores
            else:
                # Just use cross-encoder scores
                final_scores = cross_encoder_scores
            
            # Filter by threshold if needed
            if self.params["threshold"] > 0:
                selected_indices = [i for i, score in enumerate(final_scores) 
                                   if score >= self.params["threshold"]]
                if not selected_indices:  # If all filtered out, keep the best one
                    selected_indices = [np.argmax(final_scores)]
            else:
                selected_indices = list(range(len(docs)))
            
            # Create reranked document list
            reranked_docs = []
            for i in np.argsort(-np.array(final_scores)):
                if i in selected_indices:
                    # Add relevance score to metadata
                    doc = docs[i]
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["relevance_score"] = float(final_scores[i])
                    reranked_docs.append(doc)
                
                if len(reranked_docs) >= self.params["top_k"]:
                    break
            
            logger.info(f"Reranked documents: {len(docs)} -> {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Fall back to original order on error
            return docs[:self.params["top_k"]]
