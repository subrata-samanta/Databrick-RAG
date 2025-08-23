"""
Module for managing sessions for multiple users.
"""

import uuid
import time
import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

class SessionManager:
    """Class for managing sessions for multiple users"""
    
    def __init__(self, rag_pipeline, max_sessions=1000):
        """
        Initialize session manager
        
        Args:
            rag_pipeline: RAG pipeline instance
            max_sessions: Maximum number of active sessions
        """
        self.pipeline = rag_pipeline
        self.max_sessions = max_sessions
        
        # Session data storage
        self.sessions = {}
        
        # Session configuration
        self.session_config = {
            "timeout_minutes": 60,  # Session timeout in minutes
            "max_history": 20,      # Maximum conversation history items
            "cleanup_interval": 30   # Minutes between cleanup runs
        }
        
        # Last cleanup timestamp
        self.last_cleanup = time.time()
        
        logger.info(f"Session manager initialized with max {max_sessions} sessions")
    
    def create_session(self, user_id: str = None) -> str:
        """
        Create a new session
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        # Clean up expired sessions if needed
        self._maybe_cleanup_sessions()
        
        # Check if we're at max capacity
        if len(self.sessions) >= self.max_sessions:
            # Remove the oldest session
            oldest_session = min(self.sessions.items(), key=lambda x: x[1]["last_active"])
            logger.info(f"Removing oldest session {oldest_session[0]} to make room")
            del self.sessions[oldest_session[0]]
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session data
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id or "anonymous",
            "created": time.time(),
            "last_active": time.time(),
            "history": [],
            "metadata": {}
        }
        
        logger.info(f"Created session {session_id} for user {user_id or 'anonymous'}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None if not found
        """
        session = self.sessions.get(session_id)
        
        if session:
            # Update last active time
            session["last_active"] = time.time()
            return session
        
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False if session not found
        """
        if session_id in self.sessions:
            user_id = self.sessions[session_id]["user_id"]
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id} for user {user_id}")
            return True
            
        return False
    
    def add_to_history(self, session_id: str, query: str, response: Dict) -> bool:
        """
        Add query/response pair to session history
        
        Args:
            session_id: Session ID
            query: User query
            response: Response dictionary
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        # Add to history
        session["history"].append({
            "timestamp": time.time(),
            "query": query,
            "response": response
        })
        
        # Trim history if needed
        if len(session["history"]) > self.session_config["max_history"]:
            session["history"] = session["history"][-self.session_config["max_history"]:]
        
        return True
    
    def get_conversation_context(self, session_id: str, max_items: int = None) -> str:
        """
        Get conversation context from session history
        
        Args:
            session_id: Session ID
            max_items: Maximum number of history items to include
            
        Returns:
            Formatted conversation context or empty string if session not found
        """
        session = self.get_session(session_id)
        
        if not session or not session["history"]:
            return ""
        
        # Limit the number of history items
        limit = max_items or self.session_config["max_history"]
        history = session["history"][-limit:]
        
        # Format as conversation context
        context = "Previous conversation:\n"
        for item in history:
            context += f"User: {item['query']}\n"
            if isinstance(item['response'], dict) and 'response' in item['response']:
                context += f"Assistant: {item['response']['response']}\n"
            elif isinstance(item['response'], str):
                context += f"Assistant: {item['response']}\n"
            else:
                context += f"Assistant: [Complex response]\n"
        
        return context
    
    def ask(self, session_id: str, query: str) -> Dict:
        """
        Process a query in a specific session
        
        Args:
            session_id: Session ID
            query: User query
            
        Returns:
            Response dictionary
        """
        session = self.get_session(session_id)
        
        if not session:
            logger.warning(f"Attempt to use non-existent session: {session_id}")
            return {
                "status": "error",
                "response": "Session not found or expired. Please create a new session.",
                "error": "SESSION_NOT_FOUND"
            }
        
        try:
            # Get conversation context
            context = self.get_conversation_context(session_id)
            
            # Query rewriter if available
            if hasattr(self.pipeline, 'query_rewriter') and self.pipeline.query_rewriter:
                rewrite_result = self.pipeline.query_rewriter.rewrite_query(query, context)
                effective_query = rewrite_result["rewritten_query"]
            else:
                effective_query = query
            
            # Process through guardrails if available
            if hasattr(self.pipeline, 'guardrails') and self.pipeline.guardrails:
                input_safe, input_results = self.pipeline.guardrails.check_input(effective_query)
                
                if not input_safe:
                    response = {
                        "status": "blocked",
                        "response": "I'm sorry, I cannot respond to that query as it violates content policies.",
                        "query": query,
                        "guardrail_results": input_results
                    }
                    self.add_to_history(session_id, query, response)
                    return response
            
            # Process the query
            response = self.pipeline.ask(effective_query)
            
            # Apply guardrails to response if available
            if hasattr(self.pipeline, 'guardrails') and self.pipeline.guardrails and 'response' in response:
                guardrail_results = self.pipeline.guardrails.apply_guardrails(
                    query, response["response"]
                )
                response["response"] = guardrail_results["safe_response"]
                response["guardrail_results"] = guardrail_results
            
            # Add to history
            self.add_to_history(session_id, query, response)
            
            # Add session info to response
            response["session_id"] = session_id
            response["history_length"] = len(session["history"])
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query in session {session_id}: {str(e)}")
            error_response = {
                "status": "error",
                "response": "An error occurred while processing your query.",
                "error": str(e),
                "session_id": session_id
            }
            self.add_to_history(session_id, query, error_response)
            return error_response
    
    def _maybe_cleanup_sessions(self):
        """Clean up expired sessions if cleanup interval has passed"""
        current_time = time.time()
        minutes_since_cleanup = (current_time - self.last_cleanup) / 60
        
        # Skip if cleanup interval hasn't passed
        if minutes_since_cleanup < self.session_config["cleanup_interval"]:
            return
        
        logger.info("Running session cleanup")
        self.last_cleanup = current_time
        
        # Find expired sessions
        timeout_seconds = self.session_config["timeout_minutes"] * 60
        expired_sessions = [
            session_id for session_id, data in self.sessions.items()
            if (current_time - data["last_active"]) > timeout_seconds
        ]
        
        # Delete expired sessions
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        List active sessions, optionally filtered by user ID
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of session data dictionaries
        """
        # Clean up expired sessions first
        self._maybe_cleanup_sessions()
        
        if user_id:
            return [
                {
                    "id": data["id"],
                    "user_id": data["user_id"],
                    "created": data["created"],
                    "last_active": data["last_active"],
                    "history_size": len(data["history"])
                }
                for data in self.sessions.values()
                if data["user_id"] == user_id
            ]
        else:
            return [
                {
                    "id": data["id"],
                    "user_id": data["user_id"],
                    "created": data["created"],
                    "last_active": data["last_active"],
                    "history_size": len(data["history"])
                }
                for data in self.sessions.values()
            ]
