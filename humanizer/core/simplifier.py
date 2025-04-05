"""
Functions for simplifying text vocabulary.
"""
import nltk
from nltk.corpus import wordnet
from humanizer.utils.resources import ensure_nltk_resources
import random
import spacy

def simplify_vocabulary(text, nlp, word_length_threshold=8, style_trainer=None, style_strength=0.7):
    """
    Replace complex words with simpler synonyms and apply learned style.
    
    Args:
        text (str): Input text to simplify
        nlp: SpaCy language model
        word_length_threshold (int): Words longer than this will be simplified
        style_trainer: Optional StyleTrainer to apply learned style
        style_strength (float): How strongly to apply the style (0.0-1.0)
        
    Returns:
        str: Text with simplified vocabulary
    """
    ensure_nltk_resources()
    
    doc = nlp(text)
    result = text  # Start with the original text
    
    # Store replacements we'll make
    replacements = []
    
    # Track how many words we modified
    modified_count = 0
    style_applied_count = 0
    
    # First pass: identify all tokens and their potential replacements
    for token in doc:
        # Skip non-alphabetic words, stop words, and very short words
        if not token.is_alpha or token.is_stop or len(token.text) <= 3:
            continue
            
        # Different handling for complex words vs style application
        if len(token.text) > word_length_threshold:
            # This is a complex word, try to simplify it
            synonyms = get_synonyms(token.text, token.pos_)
            
            if style_trainer and style_trainer.trained:
                # Use style trainer with high strength for complex words
                simpler_word = style_trainer.get_style_appropriate_synonym(
                    token.text, 
                    synonyms,
                    strength=0.9  # Complex words get high style strength
                )
            else:
                # Use default algorithm to find simpler synonym
                simpler_word = get_simpler_synonym(token.text, token.pos_)
            
            # Only replace if we found a good synonym that's different and shorter
            if simpler_word != token.text and len(simpler_word) < len(token.text):
                # Validate grammatical fit before accepting replacement
                if is_grammatically_valid_replacement(token, simpler_word, doc):
                    replacements.append((token.idx, token.idx + len(token.text), simpler_word))
                    modified_count += 1
                
        elif style_trainer and style_trainer.trained and random.random() < style_strength:
            # This is a normal word, but we'll consider style application
            # Only continue if the word is important (noun, verb, adj, adv)
            if token.pos_ not in ('NOUN', 'VERB', 'ADJ', 'ADV'):
                continue
                
            synonyms = get_synonyms(token.text, token.pos_)
            
            # Apply style with provided strength parameter
            styled_word = style_trainer.get_style_appropriate_synonym(
                token.text,
                synonyms,
                strength=style_strength * 0.7  # Reduce strength for non-complex words
            )
            
            # Replace if the style trainer gave us something different
            if styled_word != token.text:
                # Validate grammatical fit before accepting replacement
                if is_grammatically_valid_replacement(token, styled_word, doc):
                    replacements.append((token.idx, token.idx + len(token.text), styled_word))
                    style_applied_count += 1
    
    # Sort replacements in reverse order (to avoid offset issues)
    replacements.sort(reverse=True, key=lambda x: x[0])
    
    # Apply all replacements
    for start, end, replacement in replacements:
        result = result[:start] + replacement + result[end:]
    
    return result

def is_grammatically_valid_replacement(token, replacement, doc):
    """
    Check if a replacement will maintain grammatical correctness.
    
    Args:
        token: The original token (spaCy token)
        replacement: The proposed replacement word
        doc: The spaCy doc containing the token
        
    Returns:
        bool: Whether the replacement is likely grammatically valid
    """
    # Check if parts of speech match
    if token.pos_ in ('NOUN', 'PROPN'):
        # For nouns, keep same plurality
        if token.text.endswith('s') and not replacement.endswith('s'):
            return False
    
    # For verbs, match tenses approximately
    if token.pos_ == 'VERB':
        if token.tag_.startswith('VBD') and not (replacement.endswith('ed') or replacement in irregular_past_forms):
            return False
        if token.tag_ == 'VBZ' and not replacement.endswith('s'):
            return False
        if token.tag_ == 'VBG' and not replacement.endswith('ing'):
            return False
    
    # Check for capitalization consistency
    if token.is_title and not token.is_sent_start:
        # If word is capitalized mid-sentence, replacement should maintain this
        return True
    
    return True

# Common irregular past tense verbs
irregular_past_forms = {
    'was', 'were', 'had', 'did', 'got', 'went', 'came', 'saw', 'took',
    'made', 'knew', 'thought', 'said', 'found', 'told', 'became', 'left',
    'felt', 'put', 'brought', 'began', 'kept', 'held', 'wrote', 'stood',
    'heard', 'let', 'meant', 'set', 'met', 'ran', 'paid', 'sat', 'spoke',
    'led', 'read', 'grew', 'lost', 'fell', 'sent', 'built', 'understood',
    'drew', 'broke', 'spent', 'cut', 'rose', 'drove', 'bought', 'wore',
    'chose', 'ate'
}

def get_synonyms(word, pos=None):
    """Get all valid synonyms for a word."""
    synonyms = []
    
    # Map spaCy POS tags to WordNet POS tags
    pos_map = {
        'NOUN': 'n',
        'VERB': 'v',
        'ADJ': 'a',
        'ADV': 'r'
    }
    
    # Get the WordNet POS category if available
    wordnet_pos = pos_map.get(pos, None) if pos else None
    
    # Find all synsets for the word, filtered by part of speech if provided
    if wordnet_pos:
        synsets = wordnet.synsets(word, pos=wordnet_pos)
    else:
        synsets = wordnet.synsets(word)
    
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            # Only consider the synonym if it's different from the original word
            if synonym != word and synonym.islower():
                synonyms.append(synonym)
    
    return list(set(synonyms))

def get_simpler_synonym(word, pos=None):
    """
    Find the simplest synonym for a given word.
    
    Args:
        word (str): The word to find synonyms for
        pos (str): Part of speech tag to filter appropriate synonyms
        
    Returns:
        str: The simplest appropriate synonym, or the original word if none found
    """
    synonyms = get_synonyms(word, pos)
    
    # If we have good synonyms, pick the best one
    if synonyms:
        # Filter to shorter synonyms
        shorter_synonyms = [s for s in synonyms if len(s) < len(word)]
        
        if shorter_synonyms:
            # Sort by length (shorter words are simpler)
            most_common_synonyms = sorted(shorter_synonyms, key=lambda x: len(x))
            if most_common_synonyms:
                return most_common_synonyms[0]
    
    # If no good synonym found, return the original word
    return word
