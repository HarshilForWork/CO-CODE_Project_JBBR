import streamlit as st
import time
from mcq_generator import MCQGenerator
from document_processor import DocumentProcessor

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

def main():
    st.title("PDF MCQ Generator")
    init_session_state()
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    llm = doc_processor.init_llm()
    
    if st.session_state.session_vars["mcq_generator"] is None:
        st.session_state.session_vars["mcq_generator"] = MCQGenerator(llm)
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
    if not uploaded_file:
        return

    if st.session_state.session_vars["chunks"] is None:
        with st.spinner("Processing PDF..."):
            chunks = doc_processor.process_pdf(uploaded_file.getvalue())
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