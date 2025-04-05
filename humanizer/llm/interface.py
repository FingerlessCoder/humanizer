"""
Interface for connecting to locally deployed language models.
"""
import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Dictionary of supported LLM providers and their configuration
SUPPORTED_PROVIDERS = {
    'llama.cpp': {
        'api_type': 'http',
        'default_port': 8080,
        'description': 'llama.cpp server with HTTP API',
        'url': 'https://github.com/ggerganov/llama.cpp'
    },
    'oobabooga': {
        'api_type': 'http',
        'default_port': 5000,
        'description': 'Text generation web UI by oobabooga',
        'url': 'https://github.com/oobabooga/text-generation-webui'
    },
    'gpt4all': {
        'api_type': 'python',
        'description': 'GPT4All Python bindings',
        'url': 'https://github.com/nomic-ai/gpt4all'
    },
    'ollama': {
        'api_type': 'http',
        'default_port': 11434,
        'description': 'Ollama local API server',
        'url': 'https://github.com/ollama/ollama'
    }
}

# Get user config directory for storing LLM settings
def get_config_dir():
    """Get the directory for storing LLM configuration."""
    # Base path is in the humanizer app directory
    base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    config_dir = base_path / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir

def get_available_models():
    """
    Get list of available local LLM configurations.
    
    Returns:
        list: List of available model configurations
    """
    config_dir = get_config_dir()
    config_file = config_dir / "llm_config.json"
    
    if not config_file.exists():
        # Create default config file if it doesn't exist
        default_config = {"models": []}
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        return []
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config.get("models", [])
    except Exception as e:
        print(f"Error loading LLM config: {e}")
        return []

class LLMInterface:
    """Interface for connecting to and using local LLMs."""
    
    def __init__(self, model_config=None):
        """
        Initialize the LLM interface.
        
        Args:
            model_config (dict, optional): Configuration for the model to use
        """
        self.model_config = None
        self.provider = None
        
        if model_config:
            self.load_model(model_config)
    
    def load_model(self, model_config):
        """
        Load a language model using the provided configuration.
        
        Args:
            model_config (dict): Configuration for the model
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        provider_id = model_config.get("provider")
        
        if provider_id not in SUPPORTED_PROVIDERS:
            print(f"Unsupported LLM provider: {provider_id}")
            return False
        
        self.model_config = model_config
        self.provider = provider_id
        
        # Additional initialization based on provider type
        if SUPPORTED_PROVIDERS[provider_id]["api_type"] == "python":
            try:
                if provider_id == "gpt4all":
                    # Try to import GPT4All
                    import gpt4all
                    self.gpt4all_model = gpt4all.GPT4All(model_config.get("model_path"))
            except ImportError:
                print("Error: GPT4All Python package not installed")
                print("Install it with: pip install gpt4all")
                return False
            except Exception as e:
                print(f"Error loading GPT4All model: {e}")
                return False
        
        return True
    
    def generate_text(self, prompt, max_tokens=100, temperature=0.7, 
                     top_p=0.95, stop_sequences=None, **kwargs):
        """
        Generate text from the LLM using the provided prompt.
        
        Args:
            prompt (str): The prompt to feed to the model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling (higher = more creative)
            top_p (float): Top-p sampling parameter
            stop_sequences (list): Sequences that will stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            str: The generated text
        """
        if not self.model_config or not self.provider:
            print("No LLM model loaded")
            return ""
        
        provider_id = self.provider
        
        # Handle generation based on provider type
        if SUPPORTED_PROVIDERS[provider_id]["api_type"] == "http":
            return self._generate_via_http(prompt, max_tokens, temperature, top_p, stop_sequences, **kwargs)
        elif provider_id == "gpt4all":
            return self._generate_via_gpt4all(prompt, max_tokens, temperature, top_p, stop_sequences, **kwargs)
        
        return ""
    
    def _generate_via_http(self, prompt, max_tokens, temperature, top_p, stop_sequences, **kwargs):
        """Generate text via HTTP API."""
        if not self.model_config:
            return ""
        
        api_url = self.model_config.get("api_url")
        if not api_url:
            print("API URL not configured")
            return ""
        
        provider = self.provider
        
        try:
            # Format the request based on provider
            if provider == "llama.cpp":
                payload = {
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop_sequences if stop_sequences else []
                }
                
                headers = {"Content-Type": "application/json"}
                response = requests.post(f"{api_url}/completion", 
                                        json=payload, 
                                        headers=headers)
                
                if response.status_code == 200:
                    return response.json().get("content", "")
                else:
                    print(f"API error: {response.status_code} - {response.text}")
            
            elif provider == "oobabooga":
                payload = {
                    "prompt": prompt,
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop_sequences if stop_sequences else [],
                    "preset": self.model_config.get("preset", "None")
                }
                
                headers = {"Content-Type": "application/json"}
                response = requests.post(f"{api_url}/api/v1/generate", 
                                        json=payload, 
                                        headers=headers)
                
                if response.status_code == 200:
                    return response.json().get("results", [{}])[0].get("text", "")
                else:
                    print(f"API error: {response.status_code} - {response.text}")
            
            elif provider == "ollama":
                # Fix: Correct Ollama API endpoint and payload structure
                model_name = self.model_config.get("model_name", "llama2")
                
                # Debug information
                print(f"Connecting to Ollama API at {api_url}")
                print(f"Using model: {model_name}")
                
                # Simplified payload structure for Ollama
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stream": False
                }
                
                # Add stop sequences if provided
                if stop_sequences:
                    payload["stop"] = stop_sequences
                
                headers = {"Content-Type": "application/json"}
                
                # Ollama uses /api/generate for completion requests
                endpoint = f"{api_url}/api/generate"
                print(f"Sending request to: {endpoint}")
                print(f"Payload: {json.dumps(payload, indent=2)}")
                
                response = requests.post(
                    endpoint, 
                    json=payload, 
                    headers=headers,
                    timeout=30  # Add a reasonable timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "")
                    print(f"Response received, length: {len(generated_text)} characters")
                    return generated_text
                else:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    print(error_msg)
                    return f"Error: {error_msg}"
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: Could not connect to {api_url} - {str(e)}"
            print(error_msg)
            return ""
        except Exception as e:
            error_msg = f"Error generating text via HTTP API: {str(e)}"
            print(error_msg)
            return ""
    
    def _generate_via_gpt4all(self, prompt, max_tokens, temperature, top_p, stop_sequences, **kwargs):
        """Generate text via GPT4All Python bindings."""
        if not hasattr(self, "gpt4all_model"):
            print("GPT4All model not loaded")
            return ""
        
        try:
            # Configure generation parameters
            generation_config = {
                "max_tokens": max_tokens,
                "temp": temperature,
                "top_p": top_p,
            }
            
            if stop_sequences:
                generation_config["stop"] = stop_sequences
            
            # Generate text
            response = self.gpt4all_model.generate(
                prompt, **generation_config
            )
            
            return response
        
        except Exception as e:
            print(f"Error generating text via GPT4All: {e}")
            return ""
    
    def add_model_config(self, config):
        """
        Add a new model configuration to the available models.
        
        Args:
            config (dict): The model configuration
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        if not config.get("provider") or not config.get("name"):
            print("Error: Provider and name are required")
            return False
        
        # Load existing configs
        config_dir = get_config_dir()
        config_file = config_dir / "llm_config.json"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
            else:
                file_config = {"models": []}
            
            # Check for duplicate names
            for model in file_config["models"]:
                if model.get("name") == config.get("name"):
                    print(f"Error: Model with name '{config['name']}' already exists")
                    return False
            
            # Add the new config
            file_config["models"].append(config)
            
            # Save the updated config
            with open(config_file, 'w') as f:
                json.dump(file_config, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error adding model config: {e}")
            return False
    
    def test_connection(self):
        """
        Test the connection to the LLM.
        
        Returns:
            dict: Status of the connection test
        """
        if not self.model_config or not self.provider:
            return {"success": False, "message": "No LLM model loaded"}
        
        provider_id = self.provider
        
        try:
            if SUPPORTED_PROVIDERS[provider_id]["api_type"] == "http":
                api_url = self.model_config.get("api_url")
                if not api_url:
                    return {"success": False, "message": "API URL not configured"}
                
                # Simple health check request
                if provider_id == "llama.cpp":
                    response = requests.get(f"{api_url}/health")
                    if response.status_code == 200:
                        return {"success": True, "message": "Connected successfully"}
                    else:
                        return {"success": False, "message": f"API error: {response.status_code}"}
                
                elif provider_id in ["oobabooga", "ollama"]:
                    # For Ollama, use a specific health check endpoint if available
                    if provider_id == "ollama":
                        try:
                            # Try a simple models list request first as a health check
                            response = requests.get(f"{api_url}/api/tags")
                            if response.status_code == 200:
                                model_list = response.json().get("models", [])
                                model_name = self.model_config.get("model_name", "llama2")
                                
                                # Check if the specified model exists
                                model_exists = any(m.get("name") == model_name for m in model_list)
                                if model_exists:
                                    return {"success": True, "message": f"Connected successfully. Model '{model_name}' is available."}
                                else:
                                    # If model doesn't exist in the list, it might need to be pulled
                                    return {"success": True, "message": f"Connected to Ollama, but model '{model_name}' may need to be pulled."}
                            else:
                                # Fall back to text generation test
                                print(f"Health check failed, status: {response.status_code}. Trying text generation...")
                        except Exception as e:
                            print(f"Health check error: {str(e)}. Trying text generation...")
                    
                    # Fall back to a minimal text generation as a health check
                    result = self.generate_text("Hello", max_tokens=5)
                    if result:
                        return {"success": True, "message": "Connected successfully"}
                    else:
                        return {"success": False, "message": "Failed to generate text. Check if the model is loaded properly."}
            
            elif provider_id == "gpt4all":
                if hasattr(self, "gpt4all_model"):
                    return {"success": True, "message": "Model loaded successfully"}
                else:
                    return {"success": False, "message": "Model not loaded"}
        
        except requests.exceptions.ConnectionError as e:
            return {"success": False, "message": f"Connection error: Could not connect to {api_url}. Is Ollama running?"}
        except Exception as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}
        
        return {"success": False, "message": "Unknown provider"}
