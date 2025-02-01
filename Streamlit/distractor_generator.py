from nlp_singleton import get_nlp
from typing import List

class DistractorGenerator:
    def __init__(self, vector_store):
        self.nlp = get_nlp()
        self.vector_store = vector_store
        self.similarity_threshold = 0.7

    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        """Generate more meaningful distractors."""
        candidates = []
        answer_doc = self.nlp(answer)
        answer_len = len(answer.split())

        # Strategy 1: Extract noun chunks of similar length
        doc = self.nlp(context)
        for chunk in doc.noun_chunks:
            chunk_len = len(chunk.text.split())
            if abs(chunk_len - answer_len) <= 2:  # Allow slightly larger length variation
                similarity = chunk.similarity(answer_doc)
                if 0.3 <= similarity <= 0.8:  # Slightly relaxed similarity range
                    candidates.append(chunk.text.strip())

        # Strategy 2: Use named entities of the same type
        answer_ents = answer_doc.ents
        if answer_ents:
            answer_ent_type = answer_ents[0].label_
            for ent in doc.ents:
                if (ent.label_ == answer_ent_type and 
                    ent.text.lower() != answer.lower()):
                    candidates.append(ent.text.strip())

        # Strategy 3: Use vector store for semantic similarity
        similar_chunks = self.vector_store.similarity_search(answer, k=10)  # Increase k for more candidates
        for chunk in similar_chunks:
            chunk_doc = self.nlp(chunk.page_content)
            for sent in chunk_doc.sents:
                similarity = sent.similarity(answer_doc)
                if 0.3 <= similarity <= 0.8:  # Relaxed similarity range
                    candidates.append(sent.text.strip())

        # Filter and clean candidates
        filtered_candidates = self._filter_candidates(candidates, answer)

        # If still insufficient, fallback to random sampling from context
        if len(filtered_candidates) < num_distractors:
            additional_candidates = self._fallback_distractors(doc, answer)
            filtered_candidates.extend(additional_candidates)

        return filtered_candidates[:num_distractors]

    def _filter_candidates(self, candidates: List[str], answer: str) -> List[str]:
        """Filter and clean distractor candidates."""
        filtered = []
        answer_lower = answer.lower()
        seen = set()

        for candidate in candidates:
            candidate = candidate.strip()
            candidate_lower = candidate.lower()
            
            # Skip if candidate is too similar to answer or already seen
            if (candidate_lower == answer_lower or
                candidate_lower in seen or
                self._is_substring(candidate_lower, answer_lower) or
                self._is_word_shuffle(candidate_lower, answer_lower)):
                continue
                
            seen.add(candidate_lower)
            filtered.append(candidate)

        return filtered

    def _is_substring(self, str1: str, str2: str) -> bool:
        """Check if one string is a substring of another."""
        return str1 in str2 or str2 in str1

    def _is_word_shuffle(self, str1: str, str2: str) -> bool:
        """Check if strings are just shuffled versions of same words."""
        words1 = set(str1.split())
        words2 = set(str2.split())
        return words1 == words2
    def _fallback_distractors(self, doc, answer: str) -> List[str]:
        """Fallback mechanism to generate additional distractors."""
        fallbacks = []
        for chunk in doc.noun_chunks:
            if chunk.text.lower() != answer.lower():
                fallbacks.append(chunk.text.strip())
        return fallbacks[:3]  # Limit to 3 fallback distractors