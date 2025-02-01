# main.py (streamlit app)
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
            "mcq_generator": None,
            "quiz_completed": False,
            "questions_attempted": [False] * 5,
            "question_times": [],
            "llm": None,
            "feedback": [],  # Store feedback for each question
            "difficulty_stats": {"easy": 0, "medium": 0, "hard": 0}  # Track difficulty distribution
        }

def generate_final_report(llm):
    """Generate and display an enhanced final report."""
    try:
        session_vars = st.session_state.session_vars
        total_questions = session_vars["total_questions"]
        correct_answers = session_vars["correct_answers"]
        accuracy = (correct_answers / max(total_questions, 1)) * 100
        total_time = sum(session_vars["question_times"])
        avg_time_per_question = total_time / max(total_questions, 1)

        analysis_prompt = f"""
        Please provide a brief performance analysis for a quiz with these results:
        - Total Questions: {total_questions}
        - Correct Answers: {correct_answers}
        - Accuracy: {accuracy:.1f}%
        - Average Time per Question: {avg_time_per_question:.1f} seconds
        - Total Score: {session_vars['score']}

        Focus on:
        1. Overall performance assessment
        2. Time management
        3. Two specific suggestions for improvement
        
        Keep the response concise and actionable.
        """

        # Display performance summary
        st.markdown("### 📊 Performance Summary")
        cols = st.columns(4)
        cols[0].metric("Final Score", session_vars['score'])
        cols[1].metric("Accuracy", f"{accuracy:.1f}%")
        cols[2].metric("Avg. Time", f"{avg_time_per_question:.1f}s")
        cols[3].metric("Total Time", f"{total_time:.1f}s")

        # Question attempt summary
        st.markdown("### 📝 Question Summary")
        for i, attempted in enumerate(session_vars["questions_attempted"]):
            status = "✅" if attempted else "❌"
            time_taken = session_vars["question_times"][i] if i < len(session_vars["question_times"]) else 0
            st.write(f"Question {i+1}: {status} - Time: {time_taken:.1f}s")

        # Generate AI analysis
        with st.spinner("Generating analysis..."):
            try:
                analysis = llm["model"].invoke(analysis_prompt)
                if analysis and isinstance(analysis, str):
                    st.markdown("### 🤖 Performance Analysis")
                    st.markdown(analysis)
                else:
                    raise ValueError("Invalid analysis response")
            except Exception as e:
                st.error(f"Could not generate AI analysis: {str(e)}")
                st.markdown("""
                ### Basic Analysis
                Thank you for completing the quiz! Review your performance metrics above to identify areas for improvement.
                """)

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.markdown(f"""
        ### Basic Performance Summary
        - Score: {session_vars.get('score', 0)}
        - Questions Attempted: {session_vars.get('total_questions', 0)}
        """)

        
def display_quiz_interface():
    session_vars = st.session_state.session_vars
    
    if (session_vars["current_mcq"] is None or 
        st.button("Next Question", key="next")):

        if session_vars["total_questions"] >= 5:
            session_vars["quiz_completed"] = True
            st.success("Quiz completed! Generating your report...")
            generate_final_report(session_vars["llm"])
            return

        # Generate new question
        with st.spinner("Generating question..."):
            chunks = session_vars["chunks"]
            chunk = chunks[session_vars["current_chunk_index"] % len(chunks)]
            session_vars["current_chunk_index"] = (session_vars["current_chunk_index"] + 1) % len(chunks)
            
            mcq = session_vars["mcq_generator"].generate_mcq(chunk.page_content)
            
            if mcq:
                session_vars["current_mcq"] = mcq
                session_vars["start_time"] = time.time()
                # Update difficulty stats
                difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
                difficulty = difficulty_map[mcq.difficulty_level]
                session_vars["difficulty_stats"][difficulty] += 1
            else:
                st.error("Failed to generate question. Please try again.")
                return

    # Display current question
    display_current_question()

def display_current_question():
    session_vars = st.session_state.session_vars
    mcq = session_vars["current_mcq"]
    
    if not mcq:
        return

    st.subheader(f"Question {session_vars['total_questions'] + 1}")
    st.markdown(mcq.question)
    
    # Display options with better formatting
    user_answer = st.radio(
        "Select your answer:",
        list(mcq.options.keys()),
        format_func=lambda x: f"{x}) {mcq.options[x]}"
    )

    if st.button("Submit Answer", key="submit"):
        process_answer(user_answer)

def process_answer(user_answer):
    session_vars = st.session_state.session_vars
    mcq = session_vars["current_mcq"]
    
    response_time = time.time() - session_vars["start_time"]
    session_vars["total_questions"] += 1
    session_vars["question_times"].append(response_time)
    
    # Update attempt status
    current_question_index = session_vars["total_questions"] - 1
    session_vars["questions_attempted"][current_question_index] = True
    
    if user_answer == mcq.correct_answer:
        points = max(10 - int(response_time), 2)
        session_vars["score"] += points
        session_vars["correct_answers"] += 1
        st.success(f"✅ Correct! +{points} points")
    else:
        st.error(f"❌ Incorrect. The correct answer was {mcq.correct_answer}) {mcq.options[mcq.correct_answer]}")
    
    session_vars["current_mcq"] = None

def reset_session():
    st.session_state.session_vars = None
    st.rerun()

def main():
    st.title("Interactive PDF MCQ Generator")
    init_session_state()

    try:
        # Initialize document processor and LLM
        doc_processor = DocumentProcessor()
        if st.session_state.session_vars["llm"] is None:
            st.session_state.session_vars["llm"] = doc_processor.init_llm()

        if st.session_state.session_vars["mcq_generator"] is None:
            st.session_state.session_vars["mcq_generator"] = MCQGenerator(
                st.session_state.session_vars["llm"]
            )

        uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
        if not uploaded_file:
            st.info("Please upload a PDF file to begin.")
            return

        # Process PDF
        if st.session_state.session_vars["chunks"] is None:
            with st.spinner("Processing PDF..."):
                chunks = doc_processor.process_pdf(uploaded_file.getvalue())
                st.session_state.session_vars["chunks"] = chunks

        # Display current progress
        progress = st.progress(0)
        progress.progress(min(st.session_state.session_vars["total_questions"] / 5, 1.0))

        # Main quiz interface
        if not st.session_state.session_vars["quiz_completed"]:
            display_quiz_interface()
        else:
            generate_final_report(st.session_state.session_vars["llm"])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.button("Reset Application", on_click=reset_session)

if __name__ == "__main__":
    main()