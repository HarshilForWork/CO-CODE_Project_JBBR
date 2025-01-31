import streamlit as st
import os
import tempfile
import hashlib
import time
import random
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

@dataclass
class MCQ:
    question: str
    options: Dict[str, str]
    correct_answer: str
    difficulty_level: int

# Initialize session state
def init_session_state():
    if "session_vars" not in st.session_state:
        st.session_state.session_vars = {
            "score": 0,
            "correct_answers": 0,
            "total_questions": 0,
            "current_mcq": None,
            "chunks": None,
            "start_time": time.time(),
            "mcq_pool": []  # Pool of pre-generated MCQs
        }

# Configure LLM settings once
@st.cache_resource
def init_llm():
    return {
        "embeddings": OllamaEmbeddings(model="deepseek-r1"),
        "model": OllamaLLM(
            model="qwen2.5",
            temperature=0.7,
            num_thread=4
        ),
        "vector_store": InMemoryVectorStore(OllamaEmbeddings(model="qwen2.5"))
    }

# Process PDF with caching
@st.cache_data
def process_pdf(file_content: bytes) -> List:
    file_hash = hashlib.md5(file_content).hexdigest()  # Generate a unique hash for the file
    cache_dir = "pdf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{file_hash}.pkl")
    # Check cache first
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    # Process PDF
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name
        loader = PDFPlumberLoader(temp_path)
        documents = loader.load()
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        # Cache the processed chunks
        with open(cache_path, "wb") as f:
            pickle.dump(chunks, f)
        return chunks
    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError:
                pass  # Ignore if the file is still in use

# Generate MCQ with error handling
def generate_mcq(context: str, llm: dict) -> MCQ:
    template = """
    Generate one focused MCQ from this context. Be concise and clear.
    Context: {context}
    Output format:
    Question: [Question]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [A/B/C/D]
    Difficulty Level: [1-4]
    """
    try:
        prompt = ChatPromptTemplate.from_template(template)
        response = (prompt | llm["model"]).invoke({"context": context})
        lines = response.strip().split("\n")
        # Validate the response format
        if len(lines) < 7:
            raise ValueError("Incomplete MCQ response from LLM.")
        
        question = lines[0].replace("Question: ", "")
        options = {
            "A": lines[1].replace("A) ", ""),
            "B": lines[2].replace("B) ", ""),
            "C": lines[3].replace("C) ", ""),
            "D": lines[4].replace("D) ", "")
        }
        correct_answer = lines[5].replace("Correct Answer: ", "").strip()
        difficulty_level = lines[6].replace("Difficulty Level: ", "").strip()

        # Validate fields
        if not (question and all(options.values()) and correct_answer in options and difficulty_level.isdigit()):
            raise ValueError("Malformed MCQ response from LLM.")

        return MCQ(
            question=question,
            options=options,
            correct_answer=correct_answer,
            difficulty_level=int(difficulty_level)
        )
    except Exception as e:
        st.error(f"Error generating MCQ: {e}")
        return None

# Bulk generate MCQs using parallel processing
def bulk_generate_mcqs(chunks: List, llm: dict, num_questions: int = 5) -> List[MCQ]:
    mcq_pool = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_mcq, random.choice(chunks).page_content, llm) for _ in range(num_questions)]
        for future in futures:
            mcq = future.result()
            if mcq:
                mcq_pool.append(mcq)
    return mcq_pool

def main():
    st.title("PDF MCQ Generator")
    init_session_state()
    llm = init_llm()
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
    if not uploaded_file:
        return

    if st.session_state.session_vars["chunks"] is None:
        with st.spinner("Processing PDF..."):
            st.session_state.session_vars["chunks"] = process_pdf(uploaded_file.getvalue())

    # Pre-generate MCQs if the pool is empty
    if not st.session_state.session_vars["mcq_pool"]:
        with st.spinner("Generating MCQs..."):
            st.session_state.session_vars["mcq_pool"] = bulk_generate_mcqs(
                st.session_state.session_vars["chunks"], llm, num_questions=10
            )

    # Score display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", st.session_state.session_vars["score"])
    with col2:
        st.metric("Questions", st.session_state.session_vars["total_questions"])
    with col3:
        accuracy = (st.session_state.session_vars["correct_answers"] / 
                   max(st.session_state.session_vars["total_questions"], 1)) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")

    # Generate new MCQ if needed
    if (st.session_state.session_vars["current_mcq"] is None or 
        st.button("Next Question", key="next")):
        if st.session_state.session_vars["mcq_pool"]:
            st.session_state.session_vars["current_mcq"] = st.session_state.session_vars["mcq_pool"].pop()
            st.session_state.session_vars["start_time"] = time.time()
        else:
            st.warning("No more MCQs available. Please upload another PDF.")
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