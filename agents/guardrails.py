"""
Module for implementing NeMo Guardrails for responsible AI.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class NeMoGuardrails:
    """Class implementing NeMo Guardrails for responsible AI"""
    
    def __init__(self, config):
        """
        Initialize NeMo Guardrails
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Load guardrail configuration
        self.guardrail_config = config.base_config.get("guardrails", {})
        
        # Default guardrail levels
        self.default_level = self.guardrail_config.get("default_level", "medium")
        
        # Initialize banned topics and patterns
        self.initialize_guardrails()
        
        logger.info(f"NeMo Guardrails initialized with level: {self.default_level}")
    
    def initialize_guardrails(self):
        """Initialize guardrail rules and patterns"""
        # Banned topics (simplified - would be more comprehensive in production)
        self.banned_topics = {
            "low": ["illegal activities"],
            "medium": ["illegal activities", "adult content", "violence"],
            "high": ["illegal activities", "adult content", "violence", "political extremism", 
                    "discrimination", "personal attacks"]
        }
        
        # Regex patterns for detecting sensitive content
        self.patterns = {
            "pii": re.compile(r'\b(?:\d{3}-\d{2}-\d{4}|\(\d{3}\)\s*\d{3}-\d{4}|\d{10}|\d{9})\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "profanity": re.compile(r'\b(?:damn|hell|crap|ass)\b', re.IGNORECASE)
        }
        
        # Load topic classifiers (in a real implementation, these would be ML models)
        # For now, we'll use simple keyword matching
        self.topic_keywords = {
            "illegal activities": ["hack", "steal", "illegal", "crime", "criminal"],
            "adult content": ["adult", "explicit", "nude"],
            "violence": ["kill", "attack", "violent", "weapon"],
            "political extremism": ["extremist", "radical"],
            "discrimination": ["racial", "racist", "sexist"]
        }
        
        # Define replacement templates for unsafe content
        self.replacement_templates = [
            "I cannot provide information about {topic}.",
            "I'm not able to assist with {topic}.",
            "That content violates usage policies."
        ]
    
    def check_input(self, query: str, level: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Check if input query complies with guardrail policies
        
        Args:
            query: User input query
            level: Optional override for guardrail level
            
        Returns:
            Tuple of (is_safe, details)
        """
        active_level = level or self.default_level
        
        # Results container
        results = {
            "is_safe": True,
            "violations": [],
            "pii_detected": False,
            "topics_detected": []
        }
        
        # Check for banned topics
        if active_level in self.banned_topics:
            for topic in self.banned_topics[active_level]:
                if any(keyword in query.lower() for keyword in self.topic_keywords.get(topic, [])):
                    results["is_safe"] = False
                    results["violations"].append(f"Banned topic: {topic}")
                    results["topics_detected"].append(topic)
        
        # Check for PII
        if self.patterns["pii"].search(query):
            results["is_safe"] = False
            results["violations"].append("PII detected: Possible SSN or phone number")
            results["pii_detected"] = True
        
        # Check for email addresses
        if self.patterns["email"].search(query):
            results["is_safe"] = False
            results["violations"].append("PII detected: Email address")
            results["pii_detected"] = True
        
        # Check for profanity if high level
        if active_level == "high" and self.patterns["profanity"].search(query):
            results["is_safe"] = False
            results["violations"].append("Profanity detected")
        
        # Log results if violations found
        if not results["is_safe"]:
            logger.warning(f"Guardrail input violation: {results['violations']}")
        
        return results["is_safe"], results
    
    def check_output(self, response: str, level: Optional[str] = None) -> Tuple[bool, Dict, str]:
        """
        Check if output response complies with guardrail policies
        
        Args:
            response: Generated response
            level: Optional override for guardrail level
            
        Returns:
            Tuple of (is_safe, details, safe_response)
        """
        active_level = level or self.default_level
        
        # Results container
        results = {
            "is_safe": True,
            "violations": [],
            "pii_detected": False,
            "topics_detected": []
        }
        
        # Original response
        safe_response = response
        
        # Check for banned topics in response
        if active_level in self.banned_topics:
            for topic in self.banned_topics[active_level]:
                if any(keyword in response.lower() for keyword in self.topic_keywords.get(topic, [])):
                    results["is_safe"] = False
                    results["violations"].append(f"Banned topic in response: {topic}")
                    results["topics_detected"].append(topic)
        
        # Check for PII in response
        if self.patterns["pii"].search(response):
            results["is_safe"] = False
            results["violations"].append("PII detected in response")
            results["pii_detected"] = True
            
            # Redact PII
            safe_response = self.patterns["pii"].sub("[REDACTED PII]", safe_response)
        
        # Check for email addresses in response
        if self.patterns["email"].search(response):
            results["is_safe"] = False
            results["violations"].append("Email address detected in response")
            results["pii_detected"] = True
            
            # Redact emails
            safe_response = self.patterns["email"].sub("[REDACTED EMAIL]", safe_response)
        
        # Check for profanity if high level
        if active_level == "high" and self.patterns["profanity"].search(response):
            results["is_safe"] = False
            results["violations"].append("Profanity detected in response")
            
            # Redact profanity
            safe_response = self.patterns["profanity"].sub("[REDACTED]", safe_response)
        
        # If unsafe topics detected, replace entire response
        if results["topics_detected"]:
            import random
            topic = results["topics_detected"][0]
            template = random.choice(self.replacement_templates)
            safe_response = template.format(topic=topic)
        
        # Log results if violations found
        if not results["is_safe"]:
            logger.warning(f"Guardrail output violation: {results['violations']}")
        
        return results["is_safe"], results, safe_response
    
    def apply_guardrails(self, query: str, response: str, level: Optional[str] = None) -> Dict:
        """
        Apply guardrails to input query and output response
        
        Args:
            query: User input query
            response: Generated response
            level: Optional override for guardrail level
            
        Returns:
            Dictionary with results and safe response
        """
        # Check input
        input_safe, input_results = self.check_input(query, level)
        
        # Check output only if input is safe
        if input_safe:
            output_safe, output_results, safe_response = self.check_output(response, level)
        else:
            # If input is unsafe, generate a safety message
            import random
            template = random.choice(self.replacement_templates)
            safe_response = template.format(topic="that request")
            output_safe = False
            output_results = {"is_safe": False, "violations": ["Blocked due to unsafe input"]}
        
        return {
            "original_query": query,
            "original_response": response,
            "safe_response": safe_response,
            "input_check": input_results,
            "output_check": output_results,
            "is_safe": input_safe and output_safe
        }
