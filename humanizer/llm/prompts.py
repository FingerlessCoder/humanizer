"""
Prompt templates for LLM-based text humanization.
"""

# Standard prompt for general humanization
STANDARD_HUMANIZE_PROMPT = """Please rewrite the following text to make it more human-like and natural.
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

# Prompt for avoiding AI detection
AVOID_DETECTION_PROMPT = """Rewrite the following text to bypass AI content detectors while maintaining the meaning.
Follow these guidelines:
- Use more informal language and occasionally use contractions
- Introduce minor grammatical imperfections that humans typically make
- Vary sentence lengths (include some short sentences)
- Use personal anecdotes or opinions where appropriate
- Add a few filler words occasionally (well, actually, basically)
- Include rhetorical questions occasionally
- Replace some complex vocabulary with simpler alternatives
- Make the writing style less predictable

Text to rewrite:
{text}

Human-like version that would bypass AI detection:
"""

# Prompt for academic simplification
ACADEMIC_SIMPLIFICATION_PROMPT = """Simplify the following academic or complex text to make it more readable
while preserving all key information and concepts. Your simplified version should:
- Replace jargon with simpler terms when possible
- Break down complex sentences into shorter ones
- Use active voice instead of passive voice where appropriate
- Explain concepts more clearly
- Keep specialized terms that are needed but briefly explain them
- Maintain the same logical flow and factual content
- Add examples or analogies if helpful

Complex text:
{text}

Simplified version:
"""

# Prompt for casual conversation style
CASUAL_STYLE_PROMPT = """Rewrite the following text in a casual, conversational style as if talking to a friend.
Make it sound relaxed and natural while keeping the same information and meaning:
- Use contractions (don't, can't, etc.)
- Add conversational phrases (you know, like, I mean, etc.) occasionally
- Make the tone friendly and approachable
- Use simpler vocabulary
- Include some rhetorical questions or direct address to the reader
- Break formal structures into more flowing conversation
- Keep it natural but don't make it too sloppy

Original text:
{text}

Casual conversational version:
"""

# Prompt for professional style
PROFESSIONAL_STYLE_PROMPT = """Rewrite the following text in a professional business style
that is clear, concise, and polished while maintaining all key information:
- Use professional but accessible language
- Be concise and clear
- Remove unnecessary words and redundancies
- Structure information logically 
- Maintain a confident but not overly formal tone
- Keep specialized terms where appropriate
- Ensure it sounds natural and not AI-generated

Original text:
{text}

Professional version:
"""

# Dictionary of available prompt templates
PROMPT_TEMPLATES = {
    "standard": STANDARD_HUMANIZE_PROMPT,
    "avoid_detection": AVOID_DETECTION_PROMPT,
    "academic": ACADEMIC_SIMPLIFICATION_PROMPT,
    "casual": CASUAL_STYLE_PROMPT,
    "professional": PROFESSIONAL_STYLE_PROMPT
}

def get_prompt_template(template_name="standard"):
    """
    Get a prompt template by name.
    
    Args:
        template_name (str): Name of the template to retrieve
        
    Returns:
        str: Prompt template
    """
    return PROMPT_TEMPLATES.get(template_name, STANDARD_HUMANIZE_PROMPT)

def list_available_prompts():
    """
    Get a list of available prompt templates.
    
    Returns:
        dict: Dictionary with template names and descriptions
    """
    descriptions = {
        "standard": "General humanization for more natural text",
        "avoid_detection": "Rewrite to avoid AI detection",
        "academic": "Simplify complex academic writing",
        "casual": "Casual, conversational style",
        "professional": "Clear, concise professional style",
    }
    
    return descriptions
