"""LLM client for interacting with local and remote models."""

from typing import List, Dict, Optional
from openai import OpenAI


class LLMClient:
    """Client for interacting with OpenAI-compatible LLM endpoints."""
    
    def __init__(self, base_url: str, api_key: str = "dummy", model_id: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            base_url: Base URL for the API (e.g., "http://localhost:6005/v1")
            api_key: API key (can be dummy for local servers)
            model_id: Specific model ID to use (will auto-detect if None)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id or self._get_model_id()
        self.base_url = base_url
    
    @classmethod
    def local(cls, port: int, model_id: Optional[str] = None) -> 'LLMClient':
        """Create client for local server."""
        base_url = f"http://localhost:{port}/v1"
        return cls(base_url, model_id=model_id)
    
    def _get_model_id(self) -> str:
        """Auto-detect the available model ID."""
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
            raise ValueError("No models available")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to LLM server: {e}")
    
    def prompt(
        self, 
        user_message: str, 
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send a prompt to the model and get response.
        
        Args:
            user_message: The user's prompt
            system_message: Optional system message
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model's response text
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to get response from model: {e}")
    
    def test_connection(self) -> str:
        """Test the connection to the model."""
        try:
            response = self.prompt("Hello, are you working?")
            return f"✅ Model '{self.model_id}' connected: {response[:50]}..."
        except Exception as e:
            return f"❌ Connection failed: {e}"
    
    def __repr__(self) -> str:
        return f"LLMClient(base_url='{self.base_url}', model_id='{self.model_id}')" 