"""
Module for generating synthetic data for the RAG pipeline.
"""

import uuid
import json
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DataGenerator:
    """Class for generating synthetic data for the RAG pipeline"""
    
    def __init__(self):
        """Initialize data generator"""
        self.topics = ["Machine Learning", "Data Engineering", "Cloud Computing", 
                      "Artificial Intelligence", "DevOps", "Big Data", 
                      "Business Intelligence", "Data Science", "Software Engineering"]
    
    def generate_documents(self, num_docs: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic documents for RAG pipeline
        
        Args:
            num_docs: Number of documents to generate
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for i in range(num_docs):
            # Select 1-3 random topics for this document
            doc_topics = np.random.choice(self.topics, size=np.random.randint(1, 4), replace=False)
            topic_str = ", ".join(doc_topics)
            
            # Generate document ID and title
            doc_id = str(uuid.uuid4())
            title = f"Document about {topic_str}"
            
            # Generate synthetic content based on topics
            paragraphs = []
            num_paragraphs = np.random.randint(3, 8)
            
            for _ in range(num_paragraphs):
                topic = np.random.choice(doc_topics)
                paragraph_length = np.random.randint(100, 300)
                paragraph = f"This is a synthetic paragraph about {topic}. " * (paragraph_length // 30)
                paragraphs.append(paragraph)
                
            content = "\n\n".join(paragraphs)
            
            # Add metadata
            metadata = {
                "id": doc_id,
                "title": title,
                "topics": list(doc_topics),
                "created_at": pd.Timestamp.now().isoformat(),
                "source": "synthetic",
                "word_count": len(content.split())
            }
            
            documents.append({
                "id": doc_id,
                "title": title,
                "content": content,
                "metadata": json.dumps(metadata)
            })
        
        logger.info(f"Generated {len(documents)} synthetic documents")
        return documents
    
    def generate_structured_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate synthetic structured data for SQL agent
        
        Returns:
            Dictionary containing customers, products, and orders data
        """
        # Customer data
        customer_data = []
        for i in range(100):
            customer_id = f"CUST{i:04d}"
            name = f"Customer {i}"
            segment = np.random.choice(["Enterprise", "SMB", "Consumer"])
            region = np.random.choice(["North America", "Europe", "Asia", "South America"])
            signup_date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 1000))
            
            customer_data.append({
                "customer_id": customer_id,
                "name": name,
                "segment": segment,
                "region": region,
                "signup_date": signup_date.strftime('%Y-%m-%d')
            })
        
        # Product data
        product_data = []
        for i in range(20):
            product_id = f"PROD{i:03d}"
            name = f"Product {i}"
            category = np.random.choice(["Software", "Hardware", "Service", "Subscription"])
            price = round(np.random.uniform(10, 1000), 2)
            
            product_data.append({
                "product_id": product_id,
                "name": name,
                "category": category,
                "price": price
            })
        
        # Order data
        order_data = []
        for i in range(500):
            order_id = f"ORD{i:05d}"
            customer_id = customer_data[np.random.randint(0, len(customer_data))]["customer_id"]
            product_id = product_data[np.random.randint(0, len(product_data))]["product_id"]
            quantity = np.random.randint(1, 10)
            order_date = pd.Timestamp('2021-01-01') + pd.Timedelta(days=np.random.randint(0, 730))
            
            order_data.append({
                "order_id": order_id,
                "customer_id": customer_id,
                "product_id": product_id,
                "quantity": quantity,
                "order_date": order_date.strftime('%Y-%m-%d')
            })
        
        logger.info(f"Generated synthetic structured data: {len(customer_data)} customers, {len(product_data)} products, {len(order_data)} orders")
        return {
            "customers": customer_data,
            "products": product_data,
            "orders": order_data
        }
