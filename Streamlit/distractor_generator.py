from typing import List, Optional
import random
import logging
from nltk.corpus import wordnet

class DistractorGenerator:
    def __init__(self, nlp):
        self.nlp = nlp
    
    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        strategies = [
            self._entity_based_distractors,
            self._noun_chunk_distractors,
            self._synonym_based_distractors
        ]
        
        all_distractors = set()
        for strategy in strategies:
            try:
                distractors = strategy(context, answer)
                all_distractors.update(distractors)
                if len(all_distractors) >= num_distractors:
                    break
            except Exception as e:
                logging.warning(f"Distractor strategy failed: {e}")
                continue
        
        if len(all_distractors) < num_distractors:
            all_distractors.update(self._fallback_distractors(answer))
        
        return list(all_distractors)[:num_distractors]
    
    def _entity_based_distractors(self, context: str, answer: str) -> List[str]:
        """Generate distractors based on named entities."""
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

    def _noun_chunk_distractors(self, context: str, answer: str) -> List[str]:
        """Generate distractors based on noun chunks."""
        doc = self.nlp(context)
        target_length = len(answer.split())
        return [chunk.text for chunk in doc.noun_chunks 
                if abs(len(chunk.text.split()) - target_length) <= 1
                and chunk.text.lower() != answer.lower()]

    def _synonym_based_distractors(self, context: str, answer: str) -> List[str]:
        """Generate distractors using synonyms."""
        from nltk.corpus import wordnet
        distractors = []
        for syn in wordnet.synsets(answer):
            for lemma in syn.lemmas():
                if lemma.name().lower() != answer.lower():
                    distractors.append(lemma.name())
        return distractors

    def _fallback_distractors(self, answer: str) -> List[str]:
        """Generate basic distractors as a fallback."""
        import random
        words = answer.split()
        if len(words) <= 1:
            return []
            
        distractors = []
        for _ in range(3):
            shuffled = words.copy()
            random.shuffle(shuffled)
            if shuffled != words:
                distractors.append(" ".join(shuffled))
        
        return distractors