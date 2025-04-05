"""
Main humanizer class that provides text transformation capabilities.
"""
import spacy
from language_tool_python import LanguageTool
# Change relative imports to direct imports
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
            except:
                print(f"Could not load style model from {style_model}")
    
    def train_on_texts(self, texts):
        """
        Train the humanizer on sample texts to learn writing style.
        
        Args:
            texts (list): List of text samples to learn from
        """
        if not self.style_trainer:
            self.style_trainer = StyleTrainer()
        
        self.style_trainer.train(texts)
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
        except:
            return False
    
    def humanize(self, text, simplify=True, correct=True, 
                 word_length_threshold=8, apply_style=True):
        """
        Transform text to be more human-readable.
        
        Args:
            text (str): Input text to humanize
            simplify (bool): Whether to simplify vocabulary
            correct (bool): Whether to correct grammar
            word_length_threshold (int): Words longer than this will be simplified
            apply_style (bool): Whether to apply learned style
            
        Returns:
            str: Humanized text
        """
        if not text.strip():
            return text
            
        processed_text = text
        
        # Step 1: Grammar correction
        if correct:
            processed_text = correct_grammar(processed_text, self.tool)
            
        # Step 2: Vocabulary simplification
        if simplify:
            processed_text = simplify_vocabulary(
                processed_text, 
                self.nlp, 
                word_length_threshold,
                style_trainer=self.style_trainer if apply_style else None
            )
            
        return processed_text
