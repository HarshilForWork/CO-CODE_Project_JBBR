from transformers import pipeline
import torch
import logging
import random
from typing import Optional, Dict, List
from mcq import MCQ
from topic_analyzer import TopicAnalyzer
from difficulty_analyzer import DifficultyAnalyzer
from distractor_generator import DistractorGenerator
from nlp_singleton import get_nlp

class MCQGenerator:
    def __init__(self, llm: Dict):
        self.nlp = get_nlp()
        self.llm = llm
        self.topic_analyzer = TopicAnalyzer()
        self.difficulty_analyzer = DifficultyAnalyzer(self.nlp)
        self.distractor_generator = DistractorGenerator(llm["vector_store"])
        
        # Initialize QA pipeline with error handling
        try:
            self.qa_pipeline = pipeline(
                'question-answering',
                model='distilbert-base-uncased-distilled-squad',
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logging.error(f"Failed to initialize QA pipeline: {e}")
            self.qa_pipeline = None
        
        self.option_keys = ['A', 'B', 'C', 'D']
        self.wrong_topics = []
    
    def generate_mcq(self, context: str, target_difficulty: Optional[int] = None) -> Optional[MCQ]:
        """Generate MCQ with improved error handling and logging."""
        try:
            # Generate question using LLM with more specific prompt
            question_prompt = self._get_question_template(context, target_difficulty)
            response = self.llm["model"].invoke(question_prompt)
            
            if not isinstance(response, str) or not response.strip():
                logging.error("LLM returned invalid response")
                return None
                
            question = response.strip()
            logging.info(f"Generated question: {question}")
            
            # Extract answer using QA pipeline
            if not self.qa_pipeline:
                logging.error("QA pipeline not initialized")
                return None
                
            qa_result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=50  # Limit answer length for better quality
            )
            
            correct_answer = qa_result['answer'].strip()
            if not correct_answer:
                logging.error("Failed to extract answer from context")
                return None
                
            logging.info(f"Generated answer: {correct_answer}")
            
            # Generate distractors with fallback options
            distractors = self._generate_distractors_with_fallback(context, correct_answer)
            if len(distractors) < 3:
                logging.error("Failed to generate enough distractors")
                return None
            
            # Create and validate options
            options = self._create_options(correct_answer, distractors)
            if not options:
                return None
            
            # Assess difficulty
            difficulty = self.difficulty_analyzer.assess_difficulty(
                question, correct_answer, context, qa_result['score']
            )
            
            # Extract topics
            topics = self.topic_analyzer.extract_keywords(
                context, question, correct_answer
            )
            
            return MCQ(
                question=question,
                options=options['options'],
                correct_answer=options['correct_key'],
                difficulty_level=difficulty,
                topics=topics
            )
            
        except Exception as e:
            logging.error(f"Error in MCQ generation: {str(e)}")
            return None
    
    def _generate_distractors_with_fallback(self, context: str, answer: str) -> List[str]:
        """Generate distractors with multiple fallback strategies."""
        try:
            # Try context-based distractors first
            distractors = self._extract_context_distractors(context, answer)
            
            # If we don't have enough, try keyword-based distractors
            if len(distractors) < 3:
                keyword_distractors = self._generate_keyword_distractors(context, answer)
                distractors.extend(keyword_distractors)
            
            # Final fallback: generate variations of the correct answer
            if len(distractors) < 3:
                variation_distractors = self._generate_answer_variations(answer)
                distractors.extend(variation_distractors)
            
            # Remove duplicates and similar options
            return list(set(distractors))[:3]
            
        except Exception as e:
            logging.error(f"Distractor generation failed: {e}")
            return []
    
    def _extract_context_distractors(self, context: str, answer: str) -> List[str]:
        """Extract distractors from the context."""
        doc = self.topic_analyzer.nlp(context)
        distractors = []
        
        # Extract noun phrases of similar length
        answer_length = len(answer.split())
        for chunk in doc.noun_chunks:
            if (len(chunk.text.split()) == answer_length and 
                chunk.text.lower() != answer.lower()):
                distractors.append(chunk.text)
        
        return distractors
    
    def _generate_keyword_distractors(self, context: str, answer: str) -> List[str]:
        """Generate distractors based on key terms in the context."""
        doc = self.topic_analyzer.nlp(context)
        answer_doc = self.topic_analyzer.nlp(answer)
        
        # Find sentences containing similar key terms
        answer_keywords = set(token.text.lower() for token in answer_doc 
                            if token.pos_ in ['NOUN', 'PROPN', 'VERB'])
        
        distractors = []
        for sent in doc.sents:
            sent_keywords = set(token.text.lower() for token in sent 
                              if token.pos_ in ['NOUN', 'PROPN', 'VERB'])
            
            # If there's some keyword overlap but not complete match
            if (answer_keywords & sent_keywords and 
                answer.lower() not in sent.text.lower()):
                distractors.append(sent.text.strip())
        
        return distractors
    
    def _generate_answer_variations(self, answer: str) -> List[str]:
        """Generate variations of the answer as a last resort."""
        words = answer.split()
        if len(words) <= 1:
            return []
        
        variations = []
        # Shuffle words
        for _ in range(2):
            shuffled = words.copy()
            random.shuffle(shuffled)
            if shuffled != words:
                variations.append(" ".join(shuffled))
        
        # Remove or replace a word
        if len(words) > 2:
            removed = words.copy()
            removed.pop(random.randint(0, len(removed)-1))
            variations.append(" ".join(removed))
        
        return variations
    
    def _create_options(self, correct_answer: str, distractors: List[str]) -> Optional[Dict]:
        """Create and validate MCQ options."""
        try:
            # Ensure we have enough valid distractors
            valid_distractors = [d for d in distractors 
                               if d and d.lower() != correct_answer.lower()][:3]
            
            if len(valid_distractors) < 3:
                return None
            
            # Create options dictionary
            options = {key: "" for key in self.option_keys}
            all_options = [correct_answer] + valid_distractors[:3]
            random.shuffle(all_options)
            
            # Assign options and track correct answer
            correct_key = None
            for key, option in zip(self.option_keys, all_options):
                options[key] = option
                if option == correct_answer:
                    correct_key = key
            
            return {
                "options": options,
                "correct_key": correct_key
            }
            
        except Exception as e:
            logging.error(f"Error creating options: {e}")
            return None
    
    def _get_question_template(self, context: str, target_difficulty: Optional[int] = None) -> str:
        """Get enhanced question generation template."""
        difficulty_text = {
            1: "a basic factual",
            2: "an analytical",
            3: "a complex analytical"
        }.get(target_difficulty, "an")
        
        return f"""
        Generate {difficulty_text} question based on the following text. The question should:
        1. Test understanding of key concepts
        2. Have a clear, unambiguous answer that can be found in the text
        3. Be answerable with a specific phrase or short statement
        4. Avoid yes/no or true/false questions
        
        Text: {context}
        
        Generate only the question text, without any additional explanation or context.
        """

    def record_wrong_answer(self, mcq: MCQ):
        """Record topics from wrong answers for future focus."""
        if mcq.topics:
            self.wrong_topics.extend(mcq.topics)
            # Keep only the most recent topics
            self.wrong_topics = self.wrong_topics[-10:]