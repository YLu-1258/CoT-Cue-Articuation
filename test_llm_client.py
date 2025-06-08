#!/usr/bin/env python3
"""Test script for LLM client functionality."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm.client import LLMClient


def test_model_response():
    """Test getting a response from the model."""
    print("=== Testing LLM Client ===")
    
    # Test with default local server (common port)
    try:
        # Try common local server port
        client = LLMClient.local(port=6005)
        print(f"Created client: {client}")
        
        # Test connection
        connection_result = client.test_connection()
        print(f"Connection test: {connection_result}")
        
        # Test a simple prompt
        print("\n--- Testing Simple Prompt ---")
        response = client.prompt("What is 2 + 2?")
        print(f"Question: What is 2 + 2?")
        print(f"Response: {response}")
        
        # Test with different temperature
        print("\n--- Testing with Temperature ---")
        response = client.prompt(
            "Tell me a creative story in one sentence.",
            temperature=0.7
        )
        print(f"Question: Tell me a creative story in one sentence.")
        print(f"Response: {response}")
        
        print("\n✅ All tests completed successfully!")
        
    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure your local LLM server is running on port 6005")
        print("You can try different ports by modifying the port parameter")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_custom_server(port: int):
    """Test with custom server configuration."""
    print("\n=== Testing Custom Server Configuration ===")
    
    # You can modify these values to test different servers
    custom_base_url = f"http://localhost:{port}/v1"  # Change as needed
    custom_model_id = None  # Will auto-detect
    
    try:
        client = LLMClient(base_url=custom_base_url, model_id=custom_model_id)
        print(f"Created custom client: {client}")
        
        connection_result = client.test_connection()
        print(f"Connection test: {connection_result}")
        
        if "✅" in connection_result:
            response = client.prompt("What is 2 + 2?")
            print(f"Test response: {response}")
            
    except Exception as e:
        print(f"❌ Custom server test failed: {e}")


if __name__ == "__main__":
    test_custom_server(port=6006)
    
    # print("\n=== Instructions ===")
    # print("1. Make sure your LLM server is running locally")
    # print("2. Common ports are 6005, 8000, 1234, etc.")
    # print("3. Modify the port in this script if needed")
    # print("4. Run: python test_llm_client.py") 