from typing import List, Set
import random
from Streamlit.nlp_singleton import get_nlp

class OptimizedDistractorGenerator:
    def __init__(self, vector_store):
        self.nlp = get_nlp()
        self.vector_store = vector_store
        self.max_sentence_length = 15
        self.fallback_distractors = [
            "None of the above",
            "All of the above",
            "Insufficient data to determine"
        ]
    
    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        """Generate distractors with improved fallback mechanisms."""
        candidates = set()
        
        # Try all generation methods in sequence until we have enough distractors
        generation_methods = [
            self._generate_entity_based_distractors,
            self._generate_noun_chunk_distractors,
            self._generate_vector_based_distractors,
            self._generate_syntactic_variations,
            self._get_fallback_distractors
        ]
        
        for method in generation_methods:
            new_candidates = method(context, answer)
            candidates.update(new_candidates)
            if len(candidates) >= num_distractors:
                break
        
        # Ensure we always have enough distractors
        filtered = self._filter_candidates(list(candidates), answer)
        if len(filtered) < num_distractors:
            filtered.extend(self.fallback_distractors[:num_distractors - len(filtered)])
        
        return filtered[:num_distractors]

    def _filter_candidates(self, candidates: List[str], answer: str) -> List[str]:
        """Filter and clean distractor candidates."""
        filtered = []
        answer_lower = answer.lower()
        seen = set()

        # Helper function to clean text
        def clean_text(text: str) -> str:
            return ' '.join(text.strip().split())

        # Helper function to check similarity
        def too_similar(str1: str, str2: str) -> bool:
            doc1 = self.nlp(str1)
            doc2 = self.nlp(str2)
            return doc1.similarity(doc2) > 0.85

        for candidate in candidates:
            if not candidate:
                continue
                
            candidate = clean_text(candidate)
            candidate_lower = candidate.lower()
            
            # Apply filtering criteria
            if (
                candidate_lower != answer_lower and  # Not the same as answer
                candidate_lower not in seen and      # Not duplicate
                len(candidate.split()) <= len(answer.split()) + 3 and  # Not too long
                len(candidate) >= 2 and             # Not too short
                not self._is_substring(candidate_lower, answer_lower) and  # Not substring
                not too_similar(candidate, answer)   # Not too similar
            ):
                seen.add(candidate_lower)
                filtered.append(candidate)
        
        return filtered

    def _is_substring(self, str1: str, str2: str) -> bool:
        """Check if one string is a substring of another."""
        str1_words = set(str1.split())
        str2_words = set(str2.split())
        return str1_words.issubset(str2_words) or str2_words.issubset(str1_words)
    
    def _generate_entity_based_distractors(self, context: str, answer: str) -> Set[str]:
        """Generate distractors based on named entities."""
        candidates = set()
        answer_doc = self.nlp(answer)
        context_doc = self.nlp(context)
        
        # Extract entities of same type
        if answer_doc.ents:
            answer_ent_type = answer_doc.ents[0].label_
            for ent in context_doc.ents:
                if ent.label_ == answer_ent_type and ent.text.lower() != answer.lower():
                    candidates.add(ent.text.strip())
        
        return candidates
    
    def _generate_noun_chunk_distractors(self, context: str, answer: str) -> Set[str]:
        """Generate distractors based on noun chunks."""
        candidates = set()
        doc = self.nlp(context)
        answer_len = len(answer.split())
        
        for chunk in doc.noun_chunks:
            chunk_len = len(chunk.text.split())
            if abs(chunk_len - answer_len) <= 2:
                candidates.add(chunk.text.strip())
        
        return candidates
    
    def _generate_vector_based_distractors(self, context: str, answer: str) -> Set[str]:
        """Generate distractors using vector similarity."""
        candidates = set()
        try:
            similar_chunks = self.vector_store.similarity_search(answer, k=5)
            for chunk in similar_chunks:
                candidates.add(chunk.page_content.strip())
        except Exception:
            pass
        return candidates
    
    def _generate_syntactic_variations(self, context: str, answer: str) -> Set[str]:
        """Generate distractors by creating syntactic variations."""
        candidates = set()
        doc = self.nlp(answer)
        
        # Replace key words with alternatives
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                # Get similar words based on vector similarity
                similar_words = [t.text for t in token.vocab 
                               if t.vector_norm > 0 and 
                               token.similarity(t) > 0.5][:3]
                
                for word in similar_words:
                    variation = answer.replace(token.text, word)
                    candidates.add(variation)
        
        return candidates
    
    def _get_fallback_distractors(self, context: str, answer: str) -> Set[str]:
        """Provide fallback distractors when other methods fail."""
        return set(self.fallback_distractors)