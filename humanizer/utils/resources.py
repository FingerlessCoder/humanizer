"""
Resource management utilities.
"""
import nltk
import os

def ensure_nltk_resources():
    """Ensure required NLTK resources are available."""
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('wordnet', 'corpora/wordnet'),
        ('omw-1.4', 'corpora/omw-1.4'),
    ]
    
    # Keep track of which resources we've already downloaded this session
    if not hasattr(ensure_nltk_resources, "_downloaded_resources"):
        ensure_nltk_resources._downloaded_resources = set()
    
    for resource, path in required_resources:
        # Only try to download if we haven't already downloaded it this session
        if resource not in ensure_nltk_resources._downloaded_resources:
            try:
                nltk.data.find(path)
                # Mark as downloaded to avoid checking again
                ensure_nltk_resources._downloaded_resources.add(resource)
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
                ensure_nltk_resources._downloaded_resources.add(resource)
