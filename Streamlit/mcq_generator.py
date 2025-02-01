from transformers import pipeline
import torch
import logging
import random
from typing import Optional, Dict, List
from mcq import MCQ
from topic_analyzer import TopicAnalyzer
from difficulty_analyzer import DifficultyAnalyzer
from distractor_generator import DistractorGenerator

class MCQGenerator:
    def __init__(self, llm: Dict):
        self.llm = llm
        self.topic_analyzer = TopicAnalyzer()
        self.difficulty_analyzer = DifficultyAnalyzer(self.topic_analyzer.nlp)
        self.distractor_generator = DistractorGenerator(self.topic_analyzer.nlp)
        
        # Initialize QA pipeline
        self.qa_pipeline = pipeline(
            'question-answering',
            model='distilbert-base-uncased-distilled-squad',
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.option_keys = ['A', 'B', 'C', 'D']
        self.wrong_topics = []  # Store topics user got wrong
        
    def generate_mcq(self, context: str, target_difficulty: Optional[int] = None) -> Optional[MCQ]:
        """Generate MCQ with topic awareness."""
        try:
            # Generate question using LLM
            response = self.llm["model"].invoke(
                self._get_question_template(context, target_difficulty)
            ).strip()

            # Get answer using QA pipeline
            qa_result = self.qa_pipeline(question=response, context=context)
            correct_answer = qa_result['answer'].strip()
            
            # Generate distractors
            distractors = self.distractor_generator.generate_distractors(
                context, correct_answer
            )
            
            if len(distractors) < 3:
                return None
                
            # Create options
            options = {key: "" for key in self.option_keys}
            all_options = [correct_answer] + distractors[:3]
            random.shuffle(all_options)
            
            correct_answer_key = None
            for key, option in zip(self.option_keys, all_options):
                options[key] = option
                if option == correct_answer:
                    correct_answer_key = key
            
            # Assess difficulty and extract topics
            difficulty = self.difficulty_analyzer.assess_difficulty(
                response, correct_answer, context, qa_result['score']
            )
            
            topics = self.topic_analyzer.extract_keywords(
                context, response, correct_answer
            )
            
            return MCQ(
                question=response,
                options=options,
                correct_answer=correct_answer_key,
                difficulty_level=difficulty,
                topics=topics
            )
            
        except Exception as e:
            logging.error(f"Error generating MCQ: {str(e)}")
            return None
    
    def record_wrong_answer(self, mcq: MCQ):
        """Record topics from wrong answers for future focus."""
        if mcq.topics:
            self.wrong_topics.extend(mcq.topics)
            # Keep only the most recent topics
            self.wrong_topics = self.wrong_topics[-10:]
    
    def _get_question_template(self, context: str, target_difficulty: Optional[int] = None) -> str:
        """Get question template based on difficulty and wrong topics."""
        difficulty_text = {
            1: "an easy",
            2: "a medium-difficulty",
            3: "a challenging"
        }.get(target_difficulty, "a")
        
        focus_topics = ""
        if self.wrong_topics:
            focus_topics = f" Focus on these topics if possible: {', '.join(self.wrong_topics[-3:])}"
        
        return f"""
        Based on this text, generate {difficulty_text} question that tests understanding.{focus_topics}
        
        Context: {context}

        Guidelines:
        - Make the question specific and unambiguous
        - Ensure the answer is clearly stated in the text
        - Use proper grammar and punctuation
        
        Generate only the question text, without any additional explanation.
        """