from typing import List, Optional
import logging
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import random
import re

class DistractorGenerator:
    def __init__(self, nlp):
        self.nlp = nlp
    
    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        """Generate contextually relevant distractors from the PDF content."""
        all_distractors = set()
        
        # Priority order of strategies
        strategies = [
            self._context_based_distractors,
            self._semantic_similarity_distractors,
            self._entity_based_distractors,
            self._fallback_distractors
        ]
        
        for strategy in strategies:
            try:
                distractors = strategy(context, answer)
                # Filter out distractors that are too similar to the answer
                filtered_distractors = [d for d in distractors 
                                      if self._is_valid_distractor(d, answer)]
                all_distractors.update(filtered_distractors)
                
                if len(all_distractors) >= num_distractors:
                    break
            except Exception as e:
                logging.warning(f"Distractor strategy failed: {str(e)}")
                continue
        
        return list(all_distractors)[:num_distractors]
    
    def _context_based_distractors(self, context: str, answer: str) -> List[str]:
        """Extract distractors from similar sentences in the context."""
        sentences = sent_tokenize(context)
        answer_doc = self.nlp(answer.lower())
        
        similar_phrases = []
        for sent in sentences:
            # Find phrases of similar length to the answer
            phrases = self._extract_phrases(sent)
            for phrase in phrases:
                if (len(phrase.split()) == len(answer.split()) and 
                    phrase.lower() != answer.lower()):
                    similar_phrases.append(phrase)
        
        return similar_phrases
    
    def _semantic_similarity_distractors(self, context: str, answer: str) -> List[str]:
        """Generate distractors based on semantic similarity in the context."""
        doc = self.nlp(context)
        answer_doc = self.nlp(answer)
        
        similar_phrases = []
        for sent in doc.sents:
            for chunk in sent.noun_chunks:
                if (0.3 < chunk.similarity(answer_doc) < 0.8 and 
                    chunk.text.lower() != answer.lower()):
                    similar_phrases.append(chunk.text)
        
        return similar_phrases
    
    def _entity_based_distractors(self, context: str, answer: str) -> List[str]:
        """Generate distractors based on named entities of the same type."""
        doc = self.nlp(context)
        answer_doc = self.nlp(answer)
        answer_type = next((ent.label_ for ent in answer_doc.ents), None)
        
        distractors = []
        if answer_type:
            for ent in doc.ents:
                if (ent.label_ == answer_type and 
                    ent.text.lower() != answer.lower()):
                    distractors.append(ent.text)
        
        return distractors
    
    def _fallback_distractors(self, answer: str) -> List[str]:
        """Generate contextually related but incorrect alternatives."""
        words = answer.split()
        if len(words) <= 1:
            return []
        
        distractors = []
        # Create variations by replacing key words with related terms
        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                for syn in synsets:
                    for lemma in syn.lemmas():
                        if lemma.name().lower() != word.lower():
                            new_distractor = " ".join(
                                [lemma.name() if w == word else w for w in words]
                            )
                            distractors.append(new_distractor)
        
        return distractors[:3]
    
    def _extract_phrases(self, sentence: str) -> List[str]:
        """Extract meaningful phrases from a sentence."""
        doc = self.nlp(sentence)
        phrases = []
        
        # Extract noun phrases
        phrases.extend([chunk.text for chunk in doc.noun_chunks])
        
        # Extract verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                phrase = ""
                for child in token.subtree:
                    phrase += child.text + " "
                if phrase.strip():
                    phrases.append(phrase.strip())
        
        return phrases
    
    def _is_valid_distractor(self, distractor: str, answer: str) -> bool:
        """Validate if a distractor is appropriate."""
        # Remove punctuation and convert to lowercase for comparison
        clean_distractor = re.sub(r'[^\w\s]', '', distractor.lower())
        clean_answer = re.sub(r'[^\w\s]', '', answer.lower())
        
        # Check if distractor is too similar to answer
        if clean_distractor == clean_answer:
            return False
        
        # Check if distractor is a subset or superset of answer
        if (clean_distractor in clean_answer or 
            clean_answer in clean_distractor):
            return False
        
        # Check minimum length
        if len(clean_distractor.split()) < 2:
            return False
        
        return True