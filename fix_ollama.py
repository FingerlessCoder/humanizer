"""
Diagnostic script for testing Ollama connection.
This standalone script can be used to verify if Ollama is accessible.
"""
import requests
import json
import argparse

def test_ollama(api_url="http://localhost:11434", model_name="llama2"):
    """Test connection to Ollama server and specified model."""
    print(f"Testing Ollama connection to {api_url}")
    print(f"Testing model: {model_name}")
    
    # Step 1: Check if we can connect to Ollama at all
    try:
        # List available models
        print("\n1. Checking available models...")
        response = requests.get(f"{api_url}/api/tags")
        
        if response.status_code != 200:
            print(f"  ❌ Failed to connect to Ollama: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            return False
        
        models = response.json().get("models", [])
        print(f"  ✅ Connected to Ollama successfully!")
        
        # Display available models
        if models:
            print("\n  Available models:")
            for model in models:
                print(f"    - {model.get('name')}")
        else:
            print("  No models found. You need to pull a model using 'ollama pull <model>'")
        
        # Step 2: Check if the specified model is available
        model_exists = any(m.get("name") == model_name for m in models)
        if model_exists:
            print(f"\n  ✅ Model '{model_name}' is available!")
        else:
            print(f"\n  ❌ Model '{model_name}' is not available.")
            print(f"  You may need to pull it with: ollama pull {model_name}")
            return False
        
        # Step 3: Try to generate text with the model
        print(f"\n2. Testing text generation with model '{model_name}'...")
        
        payload = {
            "model": model_name,
            "prompt": "Hello, this is a test. Please respond briefly.",
            "temperature": 0.7,
            "num_predict": 10,
            "stream": False
        }
        
        response = requests.post(
            f"{api_url}/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"  ❌ Failed to generate text: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            return False
        
        result = response.json()
        generated_text = result.get("response", "")
        
        if generated_text:
            print(f"  ✅ Successfully generated text:")
            print(f"  '{generated_text}'")
            print("\nOllama is working correctly! You can use it with the Text Humanizer.")
            return True
        else:
            print("  ❌ No text was generated.")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ❌ Connection error! Ollama server is not running or not accessible.")
        print("\nTroubleshooting steps:")
        print("1. Start Ollama with 'ollama serve' command")
        print("2. Check if Ollama is running on the default port (11434)")
        print("3. Check if there are any firewall rules blocking the connection")
        return False
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test connection to Ollama")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--model", default="deepseek-r1:8b", help="Model to test")
    
    args = parser.parse_args()
    
    test_ollama(args.url, args.model)
    
    print("\nIf you continue to have issues, verify that:")
    print("1. Ollama is installed correctly")
    print("2. You've pulled the model with 'ollama pull deepseek-r1:8b'")
    print("3. The API endpoint is correct (default: http://localhost:11434)")
