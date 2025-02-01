from dataclasses import dataclass
from typing import Dict

@dataclass
class DifficultyMetrics:
    score: float
    factors: Dict[str, float]
    level: int
    
class DifficultyAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp
    
    def assess_difficulty(self, question: str, answer: str, context: str, qa_score: float) -> int:
        """Assess question difficulty with detailed metrics."""
        doc = self.nlp(context)
        question_doc = self.nlp(question)
        
        factors = {
            'length': len(question.split()) + len(answer.split()),
            'named_entities': len([ent for ent in question_doc.ents]),
            'context_complexity': sum(1 for sent in doc.sents) / max(len(doc.text.split()), 1),
            'answer_length': len(answer.split()),
            'qa_confidence': qa_score,
            'rare_words': len([token for token in question_doc 
                             if token.is_alpha and not token.is_stop]),
            'technical_terms': len([token for token in question_doc 
                                 if token.pos_ in ['NOUN', 'PROPN'] and 
                                 not token.is_stop])
        }
        
        difficulty_score = (
            factors['length'] * 0.2 +
            factors['named_entities'] * 0.15 +
            factors['context_complexity'] * 0.15 +
            factors['answer_length'] * 0.15 +
            (1 - factors['qa_confidence']) * 0.15 +
            factors['rare_words'] * 0.1 +
            factors['technical_terms'] * 0.1
        )
        
        return 1 if difficulty_score < 1.5 else 2 if difficulty_score < 2.5 else 3