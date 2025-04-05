"""
Main humanizer class that provides text transformation capabilities.
"""
import spacy
from language_tool_python import LanguageTool
from humanizer.core.simplifier import simplify_vocabulary
from humanizer.core.grammar import correct_grammar
from humanizer.ml.style_trainer import StyleTrainer

class TextHumanizer:
    """Class for humanizing text - making it more readable and natural."""
    
    def __init__(self, language='en-US', model='en_core_web_sm', style_model=None):
        """Initialize the humanizer with language tools."""
        self.language = language
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading {model} model...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
        
        self.tool = LanguageTool(language)
        
        # Initialize style model if provided
        self.style_trainer = None
        if style_model:
            try:
                self.style_trainer = StyleTrainer.load(style_model)
            except Exception as e:
                print(f"Could not load style model from {style_model}: {e}")
    
    def train_on_texts(self, texts, continue_training=False):
        """
        Train the humanizer on sample texts to learn writing style.
        
        Args:
            texts (list): List of text samples to learn from
            continue_training (bool): Whether to continue training an existing model
                                     or start fresh
        """
        if not self.style_trainer or not continue_training:
            self.style_trainer = StyleTrainer()
        
        # Train with the continue_training flag
        self.style_trainer.train(texts, continue_training=continue_training)
        return self
        
    def save_style_model(self, filepath):
        """Save the trained style model to a file."""
        if self.style_trainer and self.style_trainer.trained:
            self.style_trainer.save(filepath)
            return True
        return False
    
    def load_style_model(self, filepath):
        """Load a trained style model from a file."""
        try:
            self.style_trainer = StyleTrainer.load(filepath)
            return True
        except Exception as e:
            print(f"Error loading style model: {e}")
            return False
    
    def humanize_with_llm(self, text, llm_interface, prompt_template=None, 
                         preserve_structure=True, max_tokens=1000, temperature=0.7):
        """
        Use an LLM to humanize text.
        
        Args:
            text (str): Input text to humanize
            llm_interface: Connected LLM interface
            prompt_template (str, optional): Custom prompt template to use
            preserve_structure (bool): Whether to preserve paragraph structure
            max_tokens (int): Maximum tokens for LLM response
            temperature (float): Temperature for generation (0.0-1.0)
            
        Returns:
            str: Humanized text from the LLM
        """
        if not text.strip():
            return text
            
        if not llm_interface:
            print("No LLM interface provided")
            return text
            
        # Use default prompt if none provided
        if not prompt_template:
            prompt_template = self._get_default_humanize_prompt()
        
        # Format the prompt with the input text
        full_prompt = prompt_template.format(text=text)
        
        # Generate text using the LLM
        try:
            humanized_text = llm_interface.generate_text(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return humanized_text.strip()
            
        except Exception as e:
            print(f"Error using LLM for humanization: {e}")
            return text
    
    def _get_default_humanize_prompt(self):
        """Get the default prompt template for humanizing text."""
        return """Please rewrite the following text to make it more human-like and natural.
Keep the same meaning and information, but:
- Use more conversational language
- Vary sentence structures
- Simplify complex terms where appropriate
- Fix any awkward phrasing
- Maintain the original tone but make it sound more natural
- Avoid unnatural repetition
- If there's code or specialized notation, keep it intact

Text to humanize:
{text}

Humanized version:
"""
    
    def humanize(self, text, simplify=True, correct=True, 
                 word_length_threshold=8, apply_style=True,
                 style_strength=0.7, use_llm=False, llm_interface=None):
        """
        Transform text to be more human-readable.
        
        Args:
            text (str): Input text to humanize
            simplify (bool): Whether to simplify vocabulary
            correct (bool): Whether to correct grammar
            word_length_threshold (int): Words longer than this will be simplified
            apply_style (bool): Whether to apply learned style
            style_strength (float): How strongly to apply the style (0.0-1.0)
            use_llm (bool): Whether to use LLM for humanization
            llm_interface: LLM interface to use if use_llm is True
            
        Returns:
            str: Humanized text
        """
        if not text.strip():
            return text
        
        # If using LLM and interface is provided, use LLM-based humanization
        if use_llm and llm_interface:
            return self.humanize_with_llm(text, llm_interface)
            
        processed_text = text
        
        # Step 1: Vocabulary simplification (first, before grammar correction)
        if simplify:
            processed_text = simplify_vocabulary(
                processed_text, 
                self.nlp, 
                word_length_threshold,
                style_trainer=self.style_trainer if apply_style else None,
                style_strength=style_strength
            )
            
        # Step 2: Grammar correction (applied AFTER vocabulary changes)
        if correct:
            processed_text = correct_grammar(processed_text, self.tool)
            
        return processed_text
