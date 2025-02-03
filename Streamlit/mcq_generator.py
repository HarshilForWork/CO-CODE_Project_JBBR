from typing import Optional, Dict, List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.models import MCQResponse
import hashlib
import json
from functools import lru_cache

class OptimizedMCQGenerator:
    def __init__(self, llm: Dict):
        self.llm = llm
        self.option_keys = ['A', 'B', 'C', 'D']
        self.question_cache = {}
        self.setup_prompts()
        
    def setup_prompts(self):
        # Optimized, more concise prompts
        self.combined_template = """
        Create a multiple-choice question from this text, difficulty level {difficulty}:
        {context}
        
        Format:
        Q: [question]
        A: [correct answer]
        D: [3 incorrect options separated by |]
        T: [2-3 topics separated by comma]
        """
        
    @lru_cache(maxsize=100)
    def _get_cached_response(self, context_hash: str, difficulty: int) -> Optional[MCQResponse]:
        """Cache responses using context hash as key."""
        return self.question_cache.get((context_hash, difficulty))

    def _hash_context(self, context: str) -> str:
        """Create a hash of the context for caching."""
        return hashlib.md5(context.encode()).hexdigest()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=4))
    def generate_mcq(self, context: str, target_difficulty: Optional[int] = None) -> Optional[MCQResponse]:
        """Generate MCQ with optimized single-prompt approach."""
        try:
            # Check cache first
            context_hash = self._hash_context(context)
            difficulty = max(1, min(3, target_difficulty or 2))
            
            cached_response = self._get_cached_response(context_hash, difficulty)
            if cached_response:
                return cached_response

            # Generate everything in one prompt
            response = self.llm["model"].invoke(
                self.combined_template.format(
                    difficulty=difficulty,
                    context=context[:300]  # Reduced context length
                )
            )

            # Parse response
            lines = response.strip().split('\n')
            question = next(line.split('Q: ')[1] for line in lines if line.startswith('Q: '))
            answer = next(line.split('A: ')[1] for line in lines if line.startswith('A: '))
            distractors = next(line.split('D: ')[1] for line in lines if line.startswith('D: ')).split('|')
            topics = next(line.split('T: ')[1] for line in lines if line.startswith('T: ')).split(',')

            # Create options
            options = self._create_options(answer.strip(), [d.strip() for d in distractors])
            correct_key = next(key for key, val in options.items() if val.strip() == answer.strip())

            # Create MCQ response
            mcq = MCQResponse(
                question=question.strip(),
                options=options,
                correct_answer=correct_key,
                difficulty_level=difficulty,
                topics=[t.strip() for t in topics]
            )

            # Cache the result
            self.question_cache[(context_hash, difficulty)] = mcq
            return mcq

        except Exception as e:
            logging.error(f"MCQ generation error: {str(e)}")
            return None

    def _create_options(self, answer: str, distractors: List[str]) -> Dict[str, str]:
        """Create randomized options dictionary."""
        import random
        options = [answer] + distractors[:3]
        random.shuffle(options)
        return {self.option_keys[i]: opt for i, opt in enumerate(options)}

    def preprocess_context(self, context: str) -> str:
        """Preprocess context to reduce token usage."""
        # Remove extra whitespace and normalize text
        context = ' '.join(context.split())
        # Take first few sentences only
        sentences = context.split('.')[:3]
        return '. '.join(sentences) + '.'

