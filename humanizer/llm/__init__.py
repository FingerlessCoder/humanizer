"""
Local LLM integration for the humanizer package.
"""
from humanizer.llm.interface import LLMInterface, get_available_models
from humanizer.llm.prompts import get_prompt_template, list_available_prompts

__all__ = [
    'LLMInterface', 
    'get_available_models',
    'get_prompt_template',
    'list_available_prompts'
]
