import streamlit as st
import os
import tempfile
import hashlib
import pickle
import logging
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from transformers import pipeline
import torch
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCQ Generator")

@dataclass
class MCQ:
    question: str
    options: Dict[str, str]
    correct_answer: str
    difficulty_level: int

class BERTMCQGenerator:
    def __init__(self):
        # Simplified pipeline initialization
        self.qa_pipeline = pipeline(
            'question-answering',
            model='distilbert-base-uncased-distilled-squad',
            device=0 if torch.cuda.is_available() else -1
        )
        self.option_keys = ['A', 'B', 'C', 'D']

    def generate_distractors(self, context: str, answer: str, num_distractors: int = 3) -> List[str]:
        sentences = context.split('.')
        distractors = []
        
        # Extract key phrases different from the answer
        for sentence in sentences:
            if len(distractors) >= num_distractors:
                break
                
            if answer.lower() in sentence.lower():
                continue
                
            words = sentence.strip().split()
            if len(words) > 2:
                potential_distractor = ' '.join(words[:3])
                if potential_distractor != answer and potential_distractor not in distractors:
                    distractors.append(potential_distractor)
        
        # Fill remaining slots with context parts
        while len(distractors) < num_distractors:
            random_start = random.randint(0, max(0, len(context.split()) - 3))
            words = context.split()[random_start:random_start + 3]
            distractor = ' '.join(words)
            if distractor != answer and distractor not in distractors:
                distractors.append(distractor)
        
        return distractors

    def assess_difficulty(self, question: str, answer: str) -> int:
        # Simplified difficulty assessment
        total_length = len(question.split()) + len(answer.split())
        if total_length < 8:
            return 1  # Easy
        elif total_length < 15:
            return 2  # Medium
        else:
            return 3  # Hard

class MCQGenerator:
    def __init__(self, llm: dict):
        self.llm = llm
        self.bert_generator = BERTMCQGenerator()
        self.template = """
        Given this text, generate a single concise question:
        {context}
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
            difficulty_level = self.bert_generator.assess_difficulty(question, correct_answer)

            mcq = MCQ(
                question=question,
                options=options,
                correct_answer=correct_answer_key,
                difficulty_level=difficulty_level
            )
            
            self.cache[cache_key] = mcq
            return mcq

        except Exception as e:
            logger.error(f"Error generating MCQ: {e}")
            return None

def init_session_state():
    if "session_vars" not in st.session_state:
        st.session_state.session_vars = {
            "score": 0,
            "correct_answers": 0,
            "total_questions": 0,
            "current_mcq": None,
            "chunks": None,
            "start_time": time.time(),
            "current_chunk_index": 0,
            "mcq_generator": None
        }

@st.cache_resource
def init_llm():
    return {
        "embeddings": OllamaEmbeddings(model="deepseek-r1:8b"),
        "model": OllamaLLM(model="qwen2.5:7b"),
        "vector_store": InMemoryVectorStore(OllamaEmbeddings(model="deepseek-r1:8b"))
    }

@st.cache_data
def process_pdf(file_content: bytes) -> List:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name
        
    try:
        loader = PDFPlumberLoader(temp_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True
        )
        return splitter.split_documents(documents)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError:
                pass

def main():
    st.title("PDF MCQ Generator")
    init_session_state()
    llm = init_llm()
    
    if st.session_state.session_vars["mcq_generator"] is None:
        st.session_state.session_vars["mcq_generator"] = MCQGenerator(llm)
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
    if not uploaded_file:
        return

    if st.session_state.session_vars["chunks"] is None:
        with st.spinner("Processing PDF..."):
            chunks = process_pdf(uploaded_file.getvalue())
            st.session_state.session_vars["chunks"] = chunks

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", st.session_state.session_vars["score"])
    with col2:
        st.metric("Questions", st.session_state.session_vars["total_questions"])
    with col3:
        accuracy = (st.session_state.session_vars["correct_answers"] / 
                   max(st.session_state.session_vars["total_questions"], 1)) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")

    # Generate new MCQ
    if (st.session_state.session_vars["current_mcq"] is None or 
        st.button("Next Question", key="next")):
        chunks = st.session_state.session_vars["chunks"]
        if chunks:
            chunk_index = st.session_state.session_vars["current_chunk_index"]
            chunk = chunks[chunk_index % len(chunks)]
            st.session_state.session_vars["current_chunk_index"] = (chunk_index + 1) % len(chunks)
            
            with st.spinner("Generating question..."):
                mcq = st.session_state.session_vars["mcq_generator"].generate_mcq(chunk.page_content)
                if mcq:
                    st.session_state.session_vars["current_mcq"] = mcq
                    st.session_state.session_vars["start_time"] = time.time()
                else:
                    st.error("Failed to generate question. Try again.")
                    return

    mcq = st.session_state.session_vars["current_mcq"]
    if not mcq:
        return

    # Display MCQ
    st.subheader(mcq.question)
    user_answer = st.radio("Select answer:", list(mcq.options.keys()), 
                          format_func=lambda x: f"{x}) {mcq.options[x]}")
    
    if st.button("Submit", key="submit"):
        response_time = time.time() - st.session_state.session_vars["start_time"]
        st.session_state.session_vars["total_questions"] += 1
        if user_answer == mcq.correct_answer:
            points = max(10 - int(response_time), 2)
            st.session_state.session_vars["score"] += points
            st.session_state.session_vars["correct_answers"] += 1
            st.success(f"✅ Correct! +{points} points")
        else:
            st.error(f"❌ Wrong! Correct answer: {mcq.correct_answer}")

if __name__ == "__main__":
    main()