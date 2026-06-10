"""
LLM Utility Functions for Rule Generation

This module provides utility functions for working with different LLM providers.
"""

import pandas as pd
from typing import Optional

try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMClient:
    """Client for interacting with different LLM providers."""
    
    def __init__(self, llm_package: str, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None):
        """
        Initialize LLM client based on provider.
        
        Args:
            llm_package: Either "ollama" or "openai"
            api_key: API key for the LLM provider
            base_url: Base URL for the LLM API (for Ollama or OpenAI-compatible APIs)
        """
        self.llm_package = llm_package
        
        if llm_package == "ollama":
            if OllamaClient is None:
                raise ImportError("Please install ollama: pip install ollama")
            self.client = OllamaClient(host=base_url)
            
        elif llm_package == "openai":
            if OpenAI is None:
                raise ImportError("Please install openai: pip install openai")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            
        else:
            raise ValueError(f"Unsupported LLM package: {llm_package}. Use 'ollama' or 'openai'")
    
    def ask_model(self, prompt: str, modelname: Optional[str] = None) -> str:
        """
        Send a prompt to the LLM and get the response.
        
        Args:
            prompt: The prompt to send to the model
            modelname: Name of the model to use
            
        Returns:
            The model's response as a string
        """
        if self.llm_package == "ollama":
            response = self.client.chat(model=modelname, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
            
        elif self.llm_package == "openai":
            model = modelname or self.client.models.list().data[0].id
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported LLM package: {self.llm_package}")


def create_result_dataframe(list_of_tuples: list, colnames: list) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a list of tuples.
    
    Args:
        list_of_tuples: List of tuples containing the data
        colnames: List of column names
        
    Returns:
        pandas DataFrame with the data
    """
    return pd.DataFrame(list_of_tuples, columns=colnames)