from typing import List, Optional
from langchain_core.vectorstores import InMemoryVectorStore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nlp_singleton import get_nlp

class DistractorGenerator:
    def __init__(self, vector_store: InMemoryVectorStore):
        self.nlp = get_nlp()
        self.vector_store = vector_store
        self.similarity_threshold = 0.7
        self.min_phrase_length = 3
        
    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        """Generate contextually relevant distractors using embeddings and NLP analysis."""
        all_distractors = []
        
        # Get answer embedding and context
        answer_doc = self.nlp(answer)
        context_doc = self.nlp(context)
        
        # Strategy 1: Use vector store similarity search
        similar_chunks = self.vector_store.similarity_search(answer, k=5)
        candidates = self._extract_candidate_phrases(similar_chunks)
        
        # Strategy 2: Extract contextually similar phrases
        noun_phrases = self._get_relevant_noun_phrases(context_doc, answer_doc)
        candidates.extend(noun_phrases)
        
        # Filter and rank candidates
        ranked_distractors = self._rank_distractors(candidates, answer_doc, context_doc)
        
        # Ensure we have enough distractors
        if len(ranked_distractors) < num_distractors:
            # Add contextually relevant terms as backup
            backup_distractors = self._generate_backup_distractors(context_doc, answer_doc)
            ranked_distractors.extend(backup_distractors)
        
        return ranked_distractors[:num_distractors]
    
    def _extract_candidate_phrases(self, chunks) -> List[str]:
        """Extract meaningful phrases from similar chunks."""
        phrases = []
        for chunk in chunks:
            doc = self.nlp(chunk.page_content)
            # Extract noun phrases and named entities
            for np in doc.noun_chunks:
                if len(np.text.split()) >= self.min_phrase_length:
                    phrases.append(np.text.strip())
            for ent in doc.ents:
                if len(ent.text.split()) >= self.min_phrase_length:
                    phrases.append(ent.text.strip())
        return list(set(phrases))
    
    def _get_relevant_noun_phrases(self, context_doc, answer_doc) -> List[str]:
        """Extract relevant noun phrases based on semantic similarity."""
        phrases = []
        answer_vector = answer_doc.vector
        
        for np in context_doc.noun_chunks:
            if len(np.text.split()) >= self.min_phrase_length:
                np_vector = np.vector
                similarity = cosine_similarity([answer_vector], [np_vector])[0][0]
                if 0.4 <= similarity <= 0.8:  # Similar but not too similar
                    phrases.append(np.text.strip())
                    
        return list(set(phrases))
    
    def _rank_distractors(self, candidates: List[str], answer_doc, context_doc) -> List[str]:
        """Rank and filter distractor candidates based on multiple criteria."""
        scored_candidates = []
        answer_vector = answer_doc.vector
        
        for candidate in candidates:
            candidate_doc = self.nlp(candidate)
            
            # Skip if too similar to answer
            if candidate.lower() == answer_doc.text.lower():
                continue
                
            # Calculate scores
            similarity_score = cosine_similarity([answer_vector], [candidate_doc.vector])[0][0]
            context_relevance = candidate_doc.similarity(context_doc)
            length_score = min(1.0, len(candidate.split()) / len(answer_doc.text.split()))
            
            # Combine scores
            total_score = (
                similarity_score * 0.4 +
                context_relevance * 0.4 +
                length_score * 0.2
            )
            
            if 0.4 <= similarity_score <= 0.8:  # Ensure reasonable similarity
                scored_candidates.append((candidate, total_score))
        
        # Sort by score and return unique distractors
        ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        return list(dict.fromkeys(c[0] for c in ranked))
    
    def _generate_backup_distractors(self, context_doc, answer_doc) -> List[str]:
        """Generate backup distractors using entity and key phrase extraction."""
        backup_distractors = []
        
        # Extract entities of similar type
        answer_entities = {ent.label_: ent.text for ent in answer_doc.ents}
        for ent in context_doc.ents:
            if (ent.label_ in answer_entities and 
                ent.text.lower() != answer_doc.text.lower()):
                backup_distractors.append(ent.text)
        
        # Extract key phrases with similar structure
        answer_pos_pattern = [token.pos_ for token in answer_doc]
        for sent in context_doc.sents:
            for i in range(len(sent) - len(answer_pos_pattern) + 1):
                span = sent[i:i + len(answer_pos_pattern)]
                if ([token.pos_ for token in span] == answer_pos_pattern and
                    span.text.lower() != answer_doc.text.lower()):
                    backup_distractors.append(span.text)
        
        return list(set(backup_distractors))