# nlp_singleton.py
import spacy
from functools import lru_cache

@lru_cache(maxsize=1)
def get_nlp():
    """Initialize spaCy NLP as a singleton."""
    try:
        return spacy.load('en_core_web_md')  # Use medium model for better vectors
    except OSError:
        # Fallback to small model if medium isn't available
        return spacy.load('en_core_web_sm')