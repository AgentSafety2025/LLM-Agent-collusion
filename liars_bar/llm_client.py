from openai import OpenAI
import anthropic
import google.generativeai as genai
import requests
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import time

# API keys and base URLs - replace with your actual keys
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyA1uQHthkhkxSze-F0SsIK6bvyBGKNUsjM"
MISTRAL_API_KEY = "Pt0Y1rWPma7CBYbF3j1UYVghig3dZurV"  
DEEPSEEK_API_KEY = "sk-e250b19b7fb445e6a8dd0ec63bfe18bc"

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)

class LLMClient:
    def __init__(self, 
                 openai_api_key: str = OPENAI_API_KEY,
                 anthropic_api_key: str = ANTHROPIC_API_KEY,
                 google_api_key: str = GOOGLE_API_KEY,
                 mistral_api_key: str = MISTRAL_API_KEY,
                 deepseek_api_key: str = DEEPSEEK_API_KEY):
        """Initialize the LLM client with multiple provider APIs

        Args:
            openai_api_key: API key for OpenAI services
            anthropic_api_key: API key for Anthropic services
            google_api_key: API key for Google services
            mistral_api_key: API key for Mistral AI services
        """
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # For Mistral we'll use direct API calls
        self.mistral_api_key = mistral_api_key
        self.mistral_base_url = "https://api.mistral.ai/v1"

        # For DeepSeek we'll use direct API calls
        self.deepseek_api_key = deepseek_api_key
        
        # Initialize Google client
        self.google_api_key = google_api_key
        genai.configure(api_key=google_api_key)
        
        # Model fallbacks - if a model isn't available, try these instead
        self.model_fallbacks = {
            # OpenAI fallbacks
            "gpt-4-turbo": ["gpt-4", "gpt-3.5-turbo"],
            "gpt-4": ["gpt-3.5-turbo"],
            # Anthropic fallbacks
            "claude-3.5-sonnet": ["claude-3-sonnet", "claude-3-haiku", "claude-instant-1.2"],
            "claude-3-opus": ["claude-3-sonnet", "claude-3-haiku", "claude-instant-1.2"],
            # Mistral fallbacks
            "mistral-large-latest": ["mistral-medium", "mistral-small"],
            "mistral-medium": ["mistral-small"],
            # Gemini fallbacks
            "gemini-1.5-pro": ["gemini-1.5-flash", "gemini-1.0-pro"],
        }
    
    def chat(self, messages: List[Dict[str, str]], model: str = "mistral-medium") -> Tuple[str, str]:
        """Interact with an LLM model using the appropriate API

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The LLM model to use

        Returns:
            tuple: (content, reasoning_content)
        """
        try:
            print(f"LLM Request to {model}: {messages}")
            
            # Try the requested model first
            content, reasoning_content = self._try_model_with_fallbacks(messages, model)
            
            print(f"LLM Response content: {content}")
            return content, reasoning_content
            
        except Exception as e:
            print(f"LLM API call error: {str(e)}")
            return "", ""
    
    def _try_model_with_fallbacks(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, str]:
        """Try to use the specified model with fallbacks if it fails
        
        Args:
            messages: List of message dictionaries
            model: The primary model to try
            
        Returns:
            tuple: (content, reasoning_content)
        """
        # First try the requested model
        try:
            return self._route_to_provider(messages, model)
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")
            
            # If that fails, try the fallbacks
            if model in self.model_fallbacks:
                for fallback_model in self.model_fallbacks[model]:
                    try:
                        print(f"Trying fallback model: {fallback_model}")
                        return self._route_to_provider(messages, fallback_model)
                    except Exception as fallback_e:
                        print(f"Error with fallback model {fallback_model}: {str(fallback_e)}")
            
            # If everything fails, raise the original exception
            raise
    
    def _route_to_provider(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, str]:
        """Route request to the appropriate provider based on model name
        
        Args:
            messages: List of message dictionaries
            model: Model name to use
            
        Returns:
            tuple: (content, reasoning_content)
        """
        if model.startswith("mistral"):
            return self._call_mistral(messages, model)
        elif model.startswith("claude"):
            return self._call_anthropic(messages, model)
        elif model.startswith("gemini"):
            return self._call_google(messages, model)
        elif model.startswith("deepseek"):
            return self._call_deepseek(messages, model)
        else:
            # Default to OpenAI for other models
            return self._call_openai(messages, model)
    
    def _call_openai(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, str]:
        """Call the OpenAI API

        Args:
            messages: List of message dictionaries
            model: The model name to use

        Returns:
            tuple: (content, reasoning_content)
        """
        # Add retry logic for rate limiting
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                
                if response.choices:
                    message = response.choices[0].message
                    content = message.content if message.content else ""
                    # OpenAI doesn't have reasoning_content by default
                    reasoning_content = ""
                    return content, reasoning_content
                
                return "", ""
            
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    # If we hit rate limits and have retries left
                    print(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # For other errors or if we're out of retries
                    raise
    
    def _call_anthropic(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, str]:
        """Call the Anthropic API with updated model names and fixed parameters
        
        Args:
            messages: List of message dictionaries
            model: The model name to use (will be mapped to current models)
            
        Returns:
            tuple: (content, reasoning_content)
        """
        # Map old model names to current model names
        model_mapping = {
            "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
            "claude-3.5-sonnet": "claude-3-5-sonnet-latest",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-instant-1.2": "claude-2.0"
        }
        
        # Get the current model name if it's in our mapping
        current_model = model_mapping.get(model, "claude-3-5-sonnet-latest")
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Create parameters dictionary without system at first
                api_params = {
                    "model": current_model,
                    "max_tokens": 1024,
                    "messages": anthropic_messages
                }
                
                # Modern Anthropic API call format - omitting system parameter entirely
                response = self.anthropic_client.messages.create(**api_params)
                
                # Extract the response content
                content = response.content[0].text if hasattr(response, "content") else ""
                
                # Anthropic doesn't have reasoning_content
                reasoning_content = ""
                
                return content, reasoning_content
                
            except Exception as e:
                print(f"Anthropic API error (attempt {attempt+1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    # If all attempts fail, just raise the exception
                    raise
        
        # This should not be reached
        return "", ""
        
    def _call_mistral(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, str]:
        """Call the Mistral AI API directly

        Args:
            messages: List of message dictionaries
            model: The model name to use (e.g., mistral-large-latest, mistral-medium, mistral-small)

        Returns:
            tuple: (content, reasoning_content)
        """
        url = f"{self.mistral_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        # Map model names if needed 
        model_mapping = {
            "mistral-medium": "mistral-medium"
        }
        
        actual_model = model_mapping.get(model, model)
        
        payload = {
            "model": actual_model,
            "messages": messages
        }
        
        # Add retry logic
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()  # Raise exception for HTTP errors
                response_data = response.json()
                
                if "choices" in response_data and response_data["choices"]:
                    message = response_data["choices"][0]["message"]
                    content = message.get("content", "")
                    # Mistral doesn't have reasoning_content
                    reasoning_content = ""
                    return content, reasoning_content
                
                return "", ""
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request error: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
    
    def _call_google(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, str]:
        """Call the Google Gemini API

        Args:
            messages: List of message dictionaries
            model: The model name to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash")

        Returns:
            tuple: (content, reasoning_content)
        """
        # Convert to Google's format
        gemini_messages = []
        
        for msg in messages:
            if msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
            # System messages are handled differently in Gemini
        
        # Determine correct model name
        gemini_model = model
        model_mapping = {
            "gemini-2.0-flash-thinking": "gemini-1.5-flash",
            "gemini-pro": "gemini-1.5-pro"
        }
        gemini_model = model_mapping.get(model, model)
        
        # Add retry logic
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                gemini_model_instance = genai.GenerativeModel(gemini_model)
                response = gemini_model_instance.generate_content(gemini_messages)
                
                content = response.text if hasattr(response, "text") else ""
                # Gemini doesn't have reasoning_content by default
                reasoning_content = ""
                
                return content, reasoning_content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error with Gemini API: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def _call_deepseek(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, str]:
        """Call the DeepSeek API
        
        Args:
            messages: List of message dictionaries
            model: The model name to use (e.g., "deepseek-r1", "deepseek-chat")
            
        Returns:
            tuple: (content, reasoning_content)
        """
        url = "https://api.deepseek.com/v1/chat/completions"  # This is the standard endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        
        # Map model names if needed
        model_mapping = {
            "deepseek-r1": "deepseek-chat",  # Adjust with actual model names
            "deepseek-coder": "deepseek-coder"
        }
        
        actual_model = model_mapping.get(model, model)
        
        payload = {
            "model": actual_model,
            "messages": messages
        }
        
        # Add retry logic
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()  # Raise exception for HTTP errors
                response_data = response.json()
                
                if "choices" in response_data and response_data["choices"]:
                    message = response_data["choices"][0]["message"]
                    content = message.get("content", "")
                    # Check if reasoning_content is available
                    reasoning_content = message.get("reasoning_content", "")
                    return content, reasoning_content
                
                return "", ""
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"DeepSeek API request error: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

# Usage example
if __name__ == "__main__":
    llm = LLMClient()
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    response = llm.chat(messages)
    print(f"Response: {response}")