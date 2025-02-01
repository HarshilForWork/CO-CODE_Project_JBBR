import spacy
import logging
from rake_nltk import Rake
from typing import List, Dict
from collections import Counter

class TopicAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.rake = Rake()
        
    def extract_keywords(self, text: str, question: str, answer: str) -> List[str]:
        """Extract keywords from the context, question and answer."""
        combined_text = f"{text} {question} {answer}"
        
        # Extract keywords using RAKE
        self.rake.extract_keywords_from_text(combined_text)
        rake_keywords = self.rake.get_ranked_phrases()[:5]
        
        # Extract named entities and important nouns using spaCy
        doc = self.nlp(combined_text)
        entities = [ent.text.lower() for ent in doc.ents]
        important_nouns = [token.text.lower() for token in doc 
                          if token.pos_ in ['NOUN', 'PROPN'] 
                          and not token.is_stop]
        
        # Combine and deduplicate keywords
        all_keywords = list(set(rake_keywords + entities + important_nouns))
        return all_keywords[:5]  # Return top 5 keywords