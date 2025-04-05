"""
Machine learning functionality for learning writing style from sample texts.
"""
import os
import pickle
import numpy as np
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import random

class StyleTrainer:
    """Learns writing style characteristics from sample texts."""
    
    def __init__(self, model='en_core_web_sm'):
        """Initialize the style trainer with required NLP models."""
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading {model} model...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
            
        self.style_profile = {
            'avg_sentence_length': 0,
            'avg_word_length': 0,
            'vocabulary_richness': 0,
            'word_freq': Counter(),
            'common_ngrams': Counter(),
            'pos_distribution': Counter()
        }
        
        self.synonyms_model = None
        self.trained = False
        self.vocab_set = set()  # Store vocabulary as a set for faster lookup
        
    def train(self, texts, continue_training=False):
        """
        Train the style model using a list of sample texts.
        
        Args:
            texts (list): List of text samples to learn from
            continue_training (bool): Whether to continue training an existing model
                                     or start fresh
        """
        if not texts:
            raise ValueError("No texts provided for training")
        
        # Keep track of the original vocabulary size for reporting
        original_vocab_size = len(self.vocab_set) if continue_training else 0
        
        # If we're not continuing training, reset the style profile
        if not continue_training:
            self.style_profile = {
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'vocabulary_richness': 0,
                'word_freq': Counter(),
                'common_ngrams': Counter(),
                'pos_distribution': Counter()
            }
            self.vocab_set = set()
            self.trained = False
        
        all_sentences = []
        all_words = []
        all_pos = []
        
        # Process each text and collect statistics
        for text in texts:
            doc = self.nlp(text)
            
            # Collect sentences
            sentences = list(doc.sents)
            all_sentences.extend(sentences)
            
            # Collect words and POS tags
            for token in doc:
                if token.is_alpha and not token.is_stop:
                    all_words.append(token.text.lower())
                    all_pos.append(token.pos_)
        
        if continue_training and self.trained:
            # When continuing training, we blend the new data with existing model
            # For avg_sentence_length, avg_word_length, and vocabulary_richness,
            # we'll calculate a weighted average based on the amount of data
            
            # Calculate existing data weight based on vocabulary size
            existing_weight = len(self.vocab_set) / (len(self.vocab_set) + len(all_words))
            new_weight = 1 - existing_weight
            
            # Update the style profile with weighted averages
            if all_sentences:
                new_avg_sent_len = np.mean([len(sent) for sent in all_sentences])
                self.style_profile['avg_sentence_length'] = (
                    existing_weight * self.style_profile['avg_sentence_length'] +
                    new_weight * new_avg_sent_len
                )
            
            if all_words:
                new_avg_word_len = np.mean([len(word) for word in all_words])
                self.style_profile['avg_word_length'] = (
                    existing_weight * self.style_profile['avg_word_length'] +
                    new_weight * new_avg_word_len
                )
                
                # Update word frequencies with new words
                for word in all_words:
                    self.style_profile['word_freq'][word] += 1
                
                # Update POS distribution
                for pos in all_pos:
                    self.style_profile['pos_distribution'][pos] += 1
                
                # Add new words to the vocabulary set
                self.vocab_set.update(word.lower() for word in all_words)
                
                # Recalculate vocabulary richness based on the updated set
                total_words = sum(self.style_profile['word_freq'].values())
                self.style_profile['vocabulary_richness'] = len(self.vocab_set) / total_words
                
        else:
            # Calculate style metrics from scratch
            if all_sentences:
                self.style_profile['avg_sentence_length'] = np.mean([len(sent) for sent in all_sentences])
            
            if all_words:
                self.style_profile['avg_word_length'] = np.mean([len(word) for word in all_words])
                self.style_profile['vocabulary_richness'] = len(set(all_words)) / len(all_words)
                self.style_profile['word_freq'] = Counter(all_words)
                self.style_profile['pos_distribution'] = Counter(all_pos)
                
                # Build vocabulary set for faster lookups
                self.vocab_set = set(word.lower() for word in all_words)
        
        # Create/update word embeddings for synonym selection
        # If continuing training, we'll concatenate all words with existing words
        all_text = " ".join(all_words)
        
        if continue_training and hasattr(self, 'synonyms_model') and self.synonyms_model:
            # For continuous training, we rebuild the model with all data
            # This could be optimized further in the future
            self._train_synonym_model(all_text)
        else:
            # Train from scratch
            self._train_synonym_model(all_text)
        
        self.trained = True
        
        # Print training summary
        if continue_training:
            new_vocab_size = len(self.vocab_set) - original_vocab_size
            print(f"Continued training with {len(texts)} new texts. Added {new_vocab_size} new words to vocabulary.")
            print(f"Total vocabulary size now: {len(self.vocab_set)} words")
        else:
            print(f"Trained on {len(texts)} texts. Vocabulary size: {len(self.vocab_set)} words.")
            
        return self
    
    def _train_synonym_model(self, text):
        """
        Train a model for synonym selection based on style.
        
        Args:
            text (str): Combined text to learn word usage from
        """
        # Create a TF-IDF vectorizer to learn word importance
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 1),
            min_df=1,  # Changed from 2 to 1 to be more flexible with small datasets
            max_features=5000
        )
        
        # If we don't have enough text, skip this step
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Use nearest neighbors to find similar words
            self.synonyms_model = {
                'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix,
                'nn_model': NearestNeighbors(
                    n_neighbors=5, 
                    algorithm='auto'
                ).fit(tfidf_matrix.toarray())
            }
        except ValueError:
            # Not enough data to train the model
            self.synonyms_model = None
    
    def get_style_appropriate_synonym(self, word, synonyms, strength=0.7):
        """
        Select the synonym that best matches the learned style.
        
        Args:
            word (str): Original word
            synonyms (list): List of possible synonyms
            strength (float): How strongly to apply the style (0.0-1.0)
            
        Returns:
            str: Most style-appropriate synonym or original word
        """
        if not self.trained or not synonyms:
            # If not trained or no synonyms, return original word
            return word
            
        word_lower = word.lower()
        
        # Calculate which synonyms appear in our vocabulary
        synonyms_in_vocab = [syn for syn in synonyms if syn.lower() in self.vocab_set]
        
        # If the word is in our vocabulary but none of its synonyms are,
        # keep the original word (respect the learned style)
        if word_lower in self.vocab_set and not synonyms_in_vocab:
            return word
            
        # If we found synonyms that are in our vocabulary, prefer those
        if synonyms_in_vocab:
            # Sort by frequency in our training data
            scored_synonyms = [(syn, self.style_profile['word_freq'].get(syn.lower(), 0)) 
                              for syn in synonyms_in_vocab]
            scored_synonyms.sort(key=lambda x: x[1], reverse=True)
            
            # Random chance to select a synonym based on strength
            if random.random() < strength:
                best_synonyms = [syn for syn, score in scored_synonyms if score > 0]
                
                # If we have highly scored synonyms
                if best_synonyms:
                    # Pick from the top 2 synonyms to maintain quality while adding variety
                    if len(best_synonyms) > 2:
                        best_synonyms = best_synonyms[:2]
                    return random.choice(best_synonyms)
                    
        # If no good synonym found or random chance didn't trigger, return original word
        return word
    
    def save(self, filepath):
        """Save the trained style model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'style_profile': self.style_profile,
                'vocab_set': self.vocab_set,
                'trained': self.trained
            }, f)
            print(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a trained style model from a file."""
        trainer = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            trainer.style_profile = data['style_profile']
            trainer.trained = data['trained']
            trainer.vocab_set = data.get('vocab_set', set())  # Handle legacy models
            print(f"Loaded model from {filepath} with {len(trainer.vocab_set)} vocabulary words")
        return trainer
