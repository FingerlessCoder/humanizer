"""
Grammar correction functionality.
"""

def correct_grammar(text, tool):
    """
    Correct grammatical errors in text.
    
    Args:
        text (str): Input text to correct
        tool: LanguageTool instance
        
    Returns:
        str: Text with corrected grammar
    """
    # Get all matches
    matches = tool.check(text)
    
    # Sort matches in reverse order (to avoid offset issues)
    matches.sort(key=lambda x: x.offset, reverse=True)
    
    # Apply corrections
    for match in matches:
        # Skip style suggestions, focus on grammar errors
        if match.ruleId.startswith('STYLE') or match.ruleId.startswith('TYPOGRAPHY'):
            continue
            
        # Apply the first replacement suggestion if available
        if match.replacements:
            start = match.offset
            end = match.offset + match.errorLength
            text = text[:start] + match.replacements[0] + text[end:]
    
    return text
