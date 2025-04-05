"""
Standalone script to run the example without requiring package installation.
This can be run directly: python run_example.py
"""
import os
import sys

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import from the humanizer package
from humanizer import TextHumanizer

def main():
    # Create a humanizer instance
    humanizer = TextHumanizer()
    
    # Example texts with complex vocabulary and grammar issues
    texts = [
        "The utilization of protracted vocabulary impedes comprehension.",
        "She don't know nothing about the sophisticated algorithms.",
        "The implementation of revolutionary methodologies necessitates innovative strategic planning.",
        "The cataclysmic ramifications of unprecedented meteorological phenomena exacerbate global concern."
    ]
    
    print("Original vs Humanized Text:\n")
    
    for text in texts:
        humanized = humanizer.humanize(text)
        print(f"Original: {text}")
        print(f"Humanized: {humanized}")
        print("-" * 80)

if __name__ == "__main__":
    main()
