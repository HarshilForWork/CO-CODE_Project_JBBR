from dataclasses import dataclass
from typing import Dict, Optional, List
import random
from transformers import pipeline
import torch
import logging
import hashlib
import spacy

@dataclass
class MCQ:
    question: str
    options: Dict[str, str]
    correct_answer: str
    difficulty_level: int

class BERTMCQGenerator:
    def __init__(self):
        self.qa_pipeline = pipeline(
            'question-answering',
            model='distilbert-base-uncased-distilled-squad',
            device=0 if torch.cuda.is_available() else -1
        )
        self.nlp = spacy.load('en_core_web_sm')
        self.option_keys = ['A', 'B', 'C', 'D']

    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        doc = self.nlp(context)
        answer_doc = self.nlp(answer)
        
        # Identify the type of answer (person, organization, date, etc.)
        answer_type = None
        for ent in answer_doc.ents:
            answer_type = ent.label_
            break
        
        distractors = []
        potential_distractors = []
        
        # If we identified the answer type, look for similar entities
        if answer_type:
            for ent in doc.ents:
                if ent.label_ == answer_type and ent.text.lower() != answer.lower():
                    potential_distractors.append(ent.text)
        
        # If we don't have enough entity-based distractors, look for similar phrases
        if len(potential_distractors) < num_distractors:
            # Find noun phrases that are similar in length to the answer
            for chunk in doc.noun_chunks:
                if abs(len(chunk.text.split()) - len(answer.split())) <= 2:
                    if chunk.text.lower() != answer.lower():
                        potential_distractors.append(chunk.text)
        
        # Shuffle and select the required number of distractors
        random.shuffle(potential_distractors)
        distractors = potential_distractors[:num_distractors]
        
        # If we still don't have enough distractors, generate some based on context
        while len(distractors) < num_distractors:
            sent = random.choice(list(doc.sents))
            words = sent.text.split()
            if len(words) > 3:
                start_idx = random.randint(0, len(words) - 3)
                distractor = ' '.join(words[start_idx:start_idx + 3])
                if distractor.lower() != answer.lower() and distractor not in distractors:
                    distractors.append(distractor)
        
        # Clean up distractors
        distractors = [d.strip() for d in distractors]
        distractors = [d for d in distractors if len(d) > 0 and d.lower() != answer.lower()]
        
        return distractors[:num_distractors]

    def assess_difficulty(self, question: str, answer: str, context: str) -> int:
        # More sophisticated difficulty assessment
        doc = self.nlp(context)
        question_doc = self.nlp(question)
        
        # Factors affecting difficulty:
        factors = {
            'length': len(question.split()) + len(answer.split()),
            'named_entities': len([ent for ent in question_doc.ents]),
            'context_complexity': sum(1 for sent in doc.sents) / len(doc.text.split()),
        }
        
        # Calculate difficulty score
        difficulty_score = (
            factors['length'] * 0.4 +
            factors['named_entities'] * 0.3 +
            factors['context_complexity'] * 0.3
        )
        
        if difficulty_score < 1.5:
            return 1  # Easy
        elif difficulty_score < 2.5:
            return 2  # Medium
        else:
            return 3  # Hard

class MCQGenerator:
    def __init__(self, llm: dict):
        self.llm = llm
        self.bert_generator = BERTMCQGenerator()
        self.template = """
        Given this text, generate a clear and specific question that tests understanding:
        Context: {context}
        
        Requirements:
        - Question should be focused and unambiguous
        - Answer should be found directly in the text
        - Avoid yes/no questions
        - Make it challenging but fair
        
        Generate only the question, without any other text.
        """
        self.cache = {}

    def generate_mcq(self, context: str) -> Optional[MCQ]:
        cache_key = hashlib.md5(context.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Generate question using LLM
            response = self.llm["model"].invoke(self.template.format(context=context))
            question = response.strip()

            # Generate answer using QA pipeline
            qa_result = self.bert_generator.qa_pipeline(
                question=question,
                context=context
            )
            correct_answer = qa_result['answer']

            # Generate distractors
            distractors = self.bert_generator.generate_distractors(context, correct_answer)

            # Ensure we have enough valid distractors
            if len(distractors) < 3:
                return None

            # Randomize options
            options = {key: "" for key in self.bert_generator.option_keys}
            all_options = [correct_answer] + distractors
            random.shuffle(all_options)
            
            # Track correct answer position
            correct_answer_key = None
            for key, option in zip(self.bert_generator.option_keys, all_options):
                options[key] = option
                if option == correct_answer:
                    correct_answer_key = key

            # Assess difficulty
            difficulty_level = self.bert_generator.assess_difficulty(question, correct_answer, context)

            mcq = MCQ(
                question=question,
                options=options,
                correct_answer=correct_answer_key,
                difficulty_level=difficulty_level
            )
            
            self.cache[cache_key] = mcq
            return mcq

        except Exception as e:
            logging.error(f"Error generating MCQ: {e}")
            return None