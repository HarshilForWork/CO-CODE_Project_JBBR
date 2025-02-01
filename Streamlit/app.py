import streamlit as st
import time
from mcq_generator import MCQGenerator
from document_processor import DocumentProcessor
from typing import List, Optional, Dict
import numpy as np
from nlp_singleton import get_nlp

def init_session_state():
    """Initialize session state variables."""
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
            "difficulty_stats": {"easy": 0, "medium": 0, "hard": 0},  # Track difficulty distribution
            "weak_topics": []  # Track weak topics based on wrong answers
        }

def generate_final_report(llm):
    """Generate an enhanced final report with detailed analysis."""
    try:
        session_vars = st.session_state.session_vars
        
        # Calculate overall metrics
        total_questions = session_vars["total_questions"]
        correct_answers = session_vars["correct_answers"]
        total_score = session_vars["score"]
        total_time = sum(session_vars["question_times"])
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Get wrong question analysis
        mcq_generator = session_vars["mcq_generator"]
        wrong_question_data = mcq_generator.get_wrong_question_analysis()
        
        # Display Final Report
        st.markdown("### 📊 Final Performance Report")
        st.write(f"**Total Questions:** {total_questions}")
        st.write(f"**Correct Answers:** {correct_answers}")
        st.write(f"**Total Points Scored:** {total_score}")
        st.write(f"**Total Time Taken:** {total_time:.2f} seconds")
        st.write(f"**Accuracy:** {accuracy:.2f}%")
        
        # Display difficulty distribution
        diff_dist = wrong_question_data['difficulty_distribution']
        if diff_dist:
            st.subheader("Difficulty Distribution of Missed Questions")
            for level, count in diff_dist.items():
                diff_text = {1: "Easy", 2: "Medium", 3: "Hard"}[level]
                st.write(f"• {diff_text}: {count} questions")
        
        # Show challenging topics
        if wrong_question_data['difficult_topics']:
            st.subheader("Challenging Topics")
            topic_data = wrong_question_data['difficult_topics']
            for topic, count in topic_data:
                st.write(f"• {topic}: {count} questions")
        
        # Generate personalized recommendations
        if wrong_question_data['total_wrong'] > 0:
            recommendations = generate_study_recommendations(
                wrong_question_data, 
                mcq_generator.topic_analyzer
            )
            st.markdown("### 📚 Personalized Study Recommendations")
            for rec in recommendations:
                st.write(f"• {rec}")
        
        # Provide a short summary
        st.markdown("### 📝 Summary")
        if accuracy >= 80:
            st.success("Great job! You have a strong understanding of the material.")
        elif accuracy >= 50:
            st.warning("Good effort! Focus on the challenging topics to improve further.")
        else:
            st.error("There's room for improvement. Review the recommendations and practice more.")
    
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

def generate_study_recommendations(wrong_data: Dict, topic_analyzer) -> List[str]:
    """Generate personalized study recommendations."""
    recommendations = []
    # Analyze keyword patterns
    if wrong_data['common_keywords']:
        keywords = [k[0] for k in wrong_data['common_keywords']]
        recommendations.append(
            f"Focus on understanding concepts related to: {', '.join(keywords)}"
        )
    # Analyze topic patterns
    if wrong_data['difficult_topics']:
        topics = [t[0] for t in wrong_data['difficult_topics']]
        recommendations.append(
            f"Review these key topics in detail: {', '.join(topics)}"
        )
    # Analyze difficulty patterns
    diff_dist = wrong_data['difficulty_distribution']
    if diff_dist:
        max_diff = max(diff_dist.items(), key=lambda x: x[1])[0]
        if max_diff == 3:
            recommendations.append(
                "Work on breaking down complex concepts into smaller, manageable parts"
            )
        elif max_diff == 2:
            recommendations.append(
                "Practice identifying relationships between different concepts"
            )
        else:
            recommendations.append(
                "Focus on strengthening fundamental concepts and terminology"
            )
    return recommendations

def display_quiz_interface():
    """Display the quiz interface with real-time metrics."""
    session_vars = st.session_state.session_vars
    
    # Real-Time Metrics Section
    st.markdown("### 📊 Real-Time Performance Metrics")
    st.write(f"**Correct Answers:** {session_vars['correct_answers']}")
    st.write(f"**Total Points Scored:** {session_vars['score']}")
    st.write(f"**Total Time Taken:** {sum(session_vars['question_times']):.2f} seconds")
    st.write(f"**Accuracy:** {((session_vars['correct_answers'] / session_vars['total_questions']) * 100):.2f}%"
             if session_vars["total_questions"] > 0 else "**Accuracy:** 0.00%")
    
    # Quiz Logic
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
    """Display the current question and process user answers."""
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
    """Process the user's answer and update session state."""
    session_vars = st.session_state.session_vars
    mcq = session_vars["current_mcq"]
    response_time = time.time() - session_vars["start_time"]
    
    # Update session variables
    session_vars["total_questions"] += 1
    session_vars["question_times"].append(response_time)
    
    # Update attempt status
    current_question_index = session_vars["total_questions"] - 1
    session_vars["questions_attempted"][current_question_index] = True
    
    if user_answer == mcq.correct_answer:
        points = max(10 - int(response_time), 2)  # Points based on response time
        session_vars["score"] += points
        session_vars["correct_answers"] += 1
        st.success(f"✅ Correct! +{points} points")
    else:
        st.error(f"❌ Incorrect. The correct answer was {mcq.correct_answer}) {mcq.options[mcq.correct_answer]}")
        # Record weak topics for future focus
        session_vars["weak_topics"].extend(mcq.topics)
        session_vars["mcq_generator"].record_wrong_answer(mcq)
    
    # Reset current MCQ for the next question
    session_vars["current_mcq"] = None

def reset_session():
    """Reset the session state."""
    st.session_state.session_vars = None
    st.rerun()

def main():
    """Main function to run the Streamlit app."""
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