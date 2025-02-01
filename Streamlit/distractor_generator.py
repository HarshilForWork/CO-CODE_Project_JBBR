from typing import List, Set
import random
from nlp_singleton import get_nlp

class OptimizedDistractorGenerator:
    def __init__(self, vector_store):
        self.nlp = get_nlp()
        self.vector_store = vector_store
        self.max_sentence_length = 15  # Maximum words for direct sentence use
        
    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        """Generate distractors with optimized performance."""
        # Quick length check to choose generation strategy
        answer_words = len(answer.split())
        if answer_words > self.max_sentence_length:
            return self._generate_phrase_based_distractors(context, answer, num_distractors)
        
        return self._generate_standard_distractors(context, answer, num_distractors)
    
    def _generate_standard_distractors(self, context: str, answer: str, num_distractors: int) -> List[str]:
        """Generate distractors for shorter answers using standard approach."""
        candidates = set()
        answer_doc = self.nlp(answer)
        
        # 1. Extract entities of same type (fast)
        if answer_doc.ents:
            answer_ent_type = answer_doc.ents[0].label_
            doc = self.nlp(context)
            for ent in doc.ents:
                if (ent.label_ == answer_ent_type and 
                    ent.text.lower() != answer.lower()):
                    candidates.add(ent.text.strip())
        
        # 2. Use noun chunks if needed (medium speed)
        if len(candidates) < num_distractors:
            doc = self.nlp(context)
            answer_len = len(answer.split())
            for chunk in doc.noun_chunks:
                chunk_len = len(chunk.text.split())
                if abs(chunk_len - answer_len) <= 2:
                    candidates.add(chunk.text.strip())
        
        # 3. Fallback to vector similarity if still needed (slower)
        if len(candidates) < num_distractors:
            similar_chunks = self.vector_store.similarity_search(answer, k=5)
            for chunk in similar_chunks:
                candidates.add(chunk.page_content.strip())
        
        # Clean and select final distractors
        filtered = self._filter_candidates(list(candidates), answer)
        return filtered[:num_distractors]
    
    def _generate_phrase_based_distractors(self, context: str, answer: str, num_distractors: int) -> List[str]:
        """Generate distractors for longer answers using phrase-based approach."""
        doc = self.nlp(context)
        candidates = set()
        
        # Extract key phrases from answer
        answer_doc = self.nlp(answer)
        key_phrases = [chunk.text for chunk in answer_doc.noun_chunks]
        
        # Find sentences with similar structure but different content
        for sent in doc.sents:
            if sent.text.lower() != answer.lower():
                sent_phrases = [chunk.text for chunk in sent.noun_chunks]
                if len(sent_phrases) == len(key_phrases):
                    candidates.add(sent.text.strip())
        
        # If we don't have enough candidates, use shortened versions
        if len(candidates) < num_distractors:
            for sent in doc.sents:
                if len(sent.text.split()) <= len(answer.split()) + 5:
                    candidates.add(sent.text.strip())
        
        filtered = self._filter_candidates(list(candidates), answer)
        return filtered[:num_distractors]
    
    def _filter_candidates(self, candidates: List[str], answer: str) -> List[str]:
        """Filter and clean distractor candidates."""
        filtered = []
        answer_lower = answer.lower()
        seen = set()
        
        for candidate in candidates:
            candidate = candidate.strip()
            candidate_lower = candidate.lower()
            
            if (candidate_lower != answer_lower and
                candidate_lower not in seen and
                not self._is_substring(candidate_lower, answer_lower)):
                seen.add(candidate_lower)
                filtered.append(candidate)
        
        # If we still don't have enough, create variations
        if len(filtered) < 3:
            filtered.extend(self._create_variations(answer))
        
        return filtered
    
    def _is_substring(self, str1: str, str2: str) -> bool:
        """Check if one string is a substring of another."""
        return str1 in str2 or str2 in str1
    
    def _create_variations(self, answer: str) -> List[str]:
        """Create variations of the answer as a last resort."""
        doc = self.nlp(answer)
        variations = []
        
        # Replace key nouns or adjectives
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ']:
                variation = answer.replace(token.text, self._get_replacement_word(token))
                variations.append(variation)
        
        return variations[:3]
    
    def _get_replacement_word(self, token) -> str:
        """Get a replacement word based on token type."""
        replacements = {
            'NOUN': ['factor', 'element', 'component', 'aspect', 'feature'],
            'ADJ': ['different', 'various', 'alternative', 'other', 'similar']
        }
        return random.choice(replacements.get(token.pos_, ['other']))