from dataclasses import dataclass
from typing import Dict, Optional, List
import random
from transformers import pipeline
import torch
import logging
import hashlib
from functools import lru_cache
import spacy
from nltk.corpus import wordnet
import nltk

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logging.warning(f"Failed to download wordnet: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class MCQ:
    question: str
    options: Dict[str, str]
    correct_answer: str
    difficulty_level: int

class BERTMCQGenerator:
    def __init__(self):
        try:
            self.qa_pipeline = pipeline(
                'question-answering',
                model='distilbert-base-uncased-distilled-squad',
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logging.error(f"Failed to initialize QA pipeline: {e}")
            raise

        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.info("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')

        self.option_keys = ['A', 'B', 'C', 'D']

    def get_synonyms(self, word: str, max_synonyms: int = 5) -> List[str]:
        """Get synonyms for a given word using WordNet with improved filtering."""
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym.lower() != word.lower():  # Avoid same word
                        synonyms.add(synonym)
                if len(synonyms) >= max_synonyms:
                    break
        except Exception as e:
            logging.warning(f"Error getting synonyms for {word}: {e}")
        return list(synonyms)[:max_synonyms]

    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        """Generate plausible distractors with improved filtering and fallback options."""
        try:
            doc = self.nlp(context)
            answer_doc = self.nlp(answer)

            # Identify answer type and key entities
            answer_type = None
            answer_entities = []
            for ent in answer_doc.ents:
                answer_type = ent.label_
                answer_entities.append(ent.text.lower())

            potential_distractors = set()

            # 1. Entity-based distractors
            if answer_type:
                for ent in doc.ents:
                    if (ent.label_ == answer_type and 
                        ent.text.lower() not in answer_entities and
                        len(ent.text.split()) <= len(answer.split()) + 2):
                        potential_distractors.add(ent.text)

            # 2. Noun chunk based distractors
            target_length = len(answer.split())
            for chunk in doc.noun_chunks:
                if (abs(len(chunk.text.split()) - target_length) <= 1 and
                    chunk.text.lower() not in answer_entities):
                    potential_distractors.add(chunk.text)

            # 3. Synonym-based distractors
            key_terms = [token.text for token in answer_doc 
                        if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]]
            for term in key_terms:
                synonyms = self.get_synonyms(term)
                for synonym in synonyms:
                    if synonym.lower() not in answer_entities:
                        potential_distractors.add(synonym)

            # Filter and clean distractors
            cleaned_distractors = []
            for dist in potential_distractors:
                cleaned = dist.strip()
                if (cleaned and 
                    cleaned.lower() != answer.lower() and
                    len(cleaned.split()) <= len(answer.split()) + 2):
                    cleaned_distractors.append(cleaned)

            # If we don't have enough distractors, generate some using fallback
            while len(cleaned_distractors) < num_distractors:
                fallback = self.generate_fallback_distractor(answer)
                if fallback and fallback not in cleaned_distractors:
                    cleaned_distractors.append(fallback)

            # Shuffle and select final distractors
            random.shuffle(cleaned_distractors)
            return cleaned_distractors[:num_distractors]

        except Exception as e:
            logging.error(f"Error generating distractors: {e}")
            return self.generate_basic_distractors(answer, num_distractors)

    def generate_basic_distractors(self, answer: str, num_needed: int) -> List[str]:
        """Generate basic distractors when other methods fail."""
        basic_distractors = []
        words = answer.split()
        
        for i in range(num_needed):
            if len(words) > 1:
                # Shuffle words
                shuffled = words.copy()
                while shuffled == words:
                    random.shuffle(shuffled)
                basic_distractors.append(" ".join(shuffled))
            else:
                # Modify single word
                modified = list(answer)
                random.shuffle(modified)
                basic_distractors.append("".join(modified))
        
        return basic_distractors

    def generate_fallback_distractor(self, answer: str) -> Optional[str]:
        """Generate a fallback distractor using simple word manipulation."""
        try:
            words = answer.split()
            if len(words) == 1:
                return None  # Skip single words
                
            # Try different manipulation strategies
            strategies = [
                lambda w: w[::-1],  # Reverse word order
                lambda w: w[1:] + [w[0]],  # Rotate words
                lambda w: w[:-1] + [w[-1]],  # Move last word to front
            ]
            
            for strategy in strategies:
                modified = strategy(words.copy())
                result = " ".join(modified)
                if result.lower() != answer.lower():
                    return result
                    
            return None
            
        except Exception as e:
            logging.warning(f"Fallback distractor generation failed: {e}")
            return None

    def assess_difficulty(self, question: str, answer: str, context: str, qa_result: dict) -> int:
        """Assess the difficulty level with improved metrics."""
        try:
            doc = self.nlp(context)
            question_doc = self.nlp(question)

            # Enhanced difficulty factors
            factors = {
                'length': len(question.split()) + len(answer.split()),
                'named_entities': len([ent for ent in question_doc.ents]),
                'context_complexity': sum(1 for sent in doc.sents) / max(len(doc.text.split()), 1),
                'answer_length': len(answer.split()),
                'qa_confidence': qa_result.get('score', 0.5),
                'rare_words': len([token for token in question_doc 
                                 if token.is_alpha and not token.is_stop]),
                'technical_terms': len([token for token in question_doc 
                                     if token.pos_ in ['NOUN', 'PROPN'] and 
                                     not token.is_stop])
            }

            # Calculate weighted difficulty score
            difficulty_score = (
                factors['length'] * 0.2 +
                factors['named_entities'] * 0.15 +
                factors['context_complexity'] * 0.15 +
                factors['answer_length'] * 0.15 +
                (1 - factors['qa_confidence']) * 0.15 +
                factors['rare_words'] * 0.1 +
                factors['technical_terms'] * 0.1
            )

            # Map score to difficulty levels
            if difficulty_score < 1.5:
                return 1  # Easy
            elif difficulty_score < 2.5:
                return 2  # Medium
            else:
                return 3  # Hard

        except Exception as e:
            logging.warning(f"Error assessing difficulty: {e}")
            return 2  # Default to medium difficulty on error

class MCQGenerator:
    def __init__(self, llm: dict):
        self.llm = llm
        try:
            self.bert_generator = BERTMCQGenerator()
        except Exception as e:
            logging.error(f"Failed to initialize BERT generator: {e}")
            raise

        self.template = """
        Based on this text, generate a clear and specific question that tests understanding:
        Context: {context}

        Guidelines:
        - Focus on key concepts or important details
        - Make the question specific and unambiguous
        - Avoid questions that can be answered without understanding the context
        - Ensure the answer is clearly stated in the text
        - Target medium difficulty level
        - Use proper grammar and punctuation

        Generate only the question text, without any additional explanation.
        """
        # Remove LRU cache and use a different approach
        self.used_questions = set()

    def generate_mcq(self, context: str) -> Optional[MCQ]:
        """Generate a unique MCQ for each call."""
        max_attempts = 3
        for _ in range(max_attempts):
            try:
                # Generate question using LLM
                response = self.llm["model"].invoke(
                    self.template.format(context=context)
                ).strip()

                # Create a unique key for this question
                question_key = hashlib.md5(response.encode()).hexdigest()

                # Skip if we've used this question before
                if question_key in self.used_questions:
                    continue

                if not response:
                    continue

                # Generate answer using QA pipeline
                qa_result = self.bert_generator.qa_pipeline(
                    question=response,
                    context=context
                )

                correct_answer = qa_result['answer'].strip()
                if not correct_answer:
                    continue

                # Generate and validate distractors
                distractors = self.bert_generator.generate_distractors(context, correct_answer)
                if len(distractors) < 3:
                    continue

                # Create options dictionary
                options = {key: "" for key in self.bert_generator.option_keys}
                all_options = [correct_answer] + distractors[:3]  # Ensure exactly 4 options
                random.shuffle(all_options)

                # Assign options and track correct answer
                correct_answer_key = None
                for key, option in zip(self.bert_generator.option_keys, all_options):
                    options[key] = option
                    if option == correct_answer:
                        correct_answer_key = key

                # Assess difficulty
                difficulty_level = self.bert_generator.assess_difficulty(
                    response, correct_answer, context, qa_result
                )

                # Add to used questions
                self.used_questions.add(question_key)

                return MCQ(
                    question=response,
                    options=options,
                    correct_answer=correct_answer_key,
                    difficulty_level=difficulty_level
                )

            except Exception as e:
                logging.error(f"Error generating MCQ: {str(e)}")
                continue

        return None
    
    
    def generate_mcq(self, context: str) -> Optional[MCQ]:
        """Generate an MCQ with caching."""
        try:
            return self.cache(context)
        except Exception as e:
            logging.error(f"Cache error in generate_mcq: {str(e)}")
            return self._generate_mcq_uncached(context)
        
    def _generate_mcq_uncached(self, context: str) -> Optional[MCQ]:
        """Generate an MCQ with improved error handling and validation."""
        try:
            # Generate question using LLM
            response = self.llm["model"].invoke(
                self.template.format(context=context)
            ).strip()

            if not response:
                raise ValueError("Empty response from LLM")

            # Generate answer using QA pipeline
            qa_result = self.bert_generator.qa_pipeline(
                question=response,
                context=context
            )

            correct_answer = qa_result['answer'].strip()
            if not correct_answer:
                raise ValueError("No valid answer extracted")

            # Generate and validate distractors
            distractors = self.bert_generator.generate_distractors(context, correct_answer)
            if len(distractors) < 3:
                raise ValueError("Insufficient valid distractors generated")

            # Create options dictionary
            options = {key: "" for key in self.bert_generator.option_keys}
            all_options = [correct_answer] + distractors
            random.shuffle(all_options)

            # Assign options and track correct answer
            correct_answer_key = None
            for key, option in zip(self.bert_generator.option_keys, all_options):
                options[key] = option
                if option == correct_answer:
                    correct_answer_key = key

            # Assess difficulty
            difficulty_level = self.bert_generator.assess_difficulty(
                response, correct_answer, context, qa_result
            )

            return MCQ(
                question=response,
                options=options,
                correct_answer=correct_answer_key,
                difficulty_level=difficulty_level
            )

        except Exception as e:
            logging.error(f"Error generating MCQ: {str(e)}")
            return None