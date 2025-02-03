# app.py
import streamlit as st
import time
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import logging
from collections import Counter
from typing import Optional, Dict
from Streamlit.performance_analyzer import PerformanceAnalyzer

BASE_URL = "http://127.0.0.1:8000"

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
            "quiz_completed": False,
            "question_times": [],
            "question_history": [],  # Track question performance
            "weak_topics": [],
            "quiz_started": False,
            "difficulty_stats": {1: 0, 2: 0, 3: 0},
            "performance_data": []  # Track performance over time
        }

def create_performance_charts():
    data = st.session_state.session_vars
    
    # Accuracy over time
    fig_accuracy = go.Figure()
    fig_accuracy.add_trace(go.Scatter(
        x=list(range(1, len(data["performance_data"]) + 1)),
        y=[d["accuracy"] for d in data["performance_data"]],
        mode='lines+markers',
        name='Accuracy'
    ))
    fig_accuracy.update_layout(
        title="Accuracy Progression",
        xaxis_title="Question Number",
        yaxis_title="Accuracy (%)"
    )
    
    # Response times
    fig_time = go.Figure()
    fig_time.add_trace(go.Box(
        y=data["question_times"],
        name="Response Times"
    ))
    fig_time.update_layout(
        title="Response Time Distribution",
        yaxis_title="Time (seconds)"
    )
    
    # Difficulty distribution
    fig_diff = px.pie(
        values=list(data["difficulty_stats"].values()),
        names=["Easy", "Medium", "Hard"],
        title="Question Difficulty Distribution"
    )
    
    return fig_accuracy, fig_time, fig_diff

def display_performance_report():
    """Display comprehensive performance report with weak points analysis."""
    analyzer = PerformanceAnalyzer()
    data = st.session_state.session_vars
    
    # Get enhanced report data
    report = analyzer.generate_report_data(data)
    
    # Display overall metrics
    st.markdown("## 📊 Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{report['metrics']['accuracy']}%")
    with col2:
        st.metric("Total Score", report['metrics']['total_score'])
    with col3:
        st.metric("Avg Response Time", f"{report['metrics']['avg_response_time']}s")
    with col4:
        st.metric("Performance Rating", report['metrics']['performance_rating'])
    
    # Display charts in tabs
    tab1, tab2, tab3 = st.tabs(["Performance", "Weak Points", "Recommendations"])
    
    with tab1:
        st.plotly_chart(report['charts']['accuracy_chart'], use_container_width=True)
        st.plotly_chart(report['charts']['time_chart'], use_container_width=True)
        st.plotly_chart(report['charts']['difficulty_chart'], use_container_width=True)
    
    with tab2:
        # Display weak points analysis
        st.markdown("### 🎯 Areas Needing Improvement")
        for topic in report['weak_points']['topics']:
            st.progress(topic['percentage'] / 100)
            st.markdown(f"**{topic['topic']}**: {topic['percentage']:.1f}% error rate")
        
        # Display error patterns
        st.markdown("### 📉 Error Patterns")
        patterns = report['weak_points']['patterns']
        
        if patterns['time_related']['found']:
            st.markdown("#### Time-based Errors")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Errors in Slow Responses", patterns['time_related']['slow_response_errors'])
            with col2:
                st.metric("Errors in Fast Responses", patterns['time_related']['fast_response_errors'])
        
        if patterns['consecutive_errors']['found']:
            st.markdown("#### Error Streaks")
            st.metric("Longest Error Streak", patterns['consecutive_errors']['max_streak'])
    
    with tab3:
        # Display recommendations
        st.markdown("### 💡 Recommendations")
        for i, rec in enumerate(report['weak_points']['recommendations'], 1):
            st.markdown(f"{i}. {rec}")
        
        # Display learning path
        if report['learning_path']:
            st.markdown("### 📚 Suggested Learning Path")
            for path in report['learning_path']:
                with st.expander(f"{path['topic']} (Priority: {path['priority']})"):
                    for focus in path['focus_areas']:
                        st.markdown(f"- {focus}")
    
    # Add an action button for the next steps
    if st.button("Start New Quiz"):
        reset_quiz()

def display_quiz_interface():
    st.markdown("### 📝 Current Quiz Progress")
    
    # Show progress bar
    progress = st.session_state.session_vars["total_questions"] / 5
    st.progress(progress)
    
    # Display current stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions", f"{st.session_state.session_vars['total_questions']}/5")
    with col2:
        st.metric("Score", st.session_state.session_vars["score"])
    with col3:
        accuracy = (st.session_state.session_vars["correct_answers"] / 
                   max(st.session_state.session_vars["total_questions"], 1) * 100)
        st.metric("Accuracy", f"{accuracy:.1f}%")

    if st.session_state.session_vars["quiz_completed"]:
        display_performance_report()
        if st.button("Start New Quiz"):
            reset_quiz()
        return

    if st.session_state.session_vars["current_mcq"] is None or st.button("Next Question", key="next_q"):
        load_next_question()
    
    display_current_question()

def load_next_question():
    if st.session_state.session_vars["total_questions"] >= 5:
        st.session_state.session_vars["quiz_completed"] = True
        return

    chunks = st.session_state.session_vars["chunks"]
    if not chunks:
        st.error("No content available. Please upload a PDF first.")
        return

    chunk = chunks[st.session_state.session_vars["current_chunk_index"] % len(chunks)]
    st.session_state.session_vars["current_chunk_index"] += 1
    
    with st.spinner("Generating question..."):
        mcq = fetch_mcq(chunk["page_content"])
        if mcq:
            st.session_state.session_vars["current_mcq"] = mcq
            st.session_state.session_vars["start_time"] = time.time()
        else:
            st.error("Failed to generate question. Please try again.")

def display_current_question():
    mcq = st.session_state.session_vars["current_mcq"]
    if not mcq:
        return

    st.markdown(f"### Question {st.session_state.session_vars['total_questions'] + 1}")
    st.markdown(mcq["question"])

    # Display options with better formatting
    selected_answer = st.radio(
        "Select your answer:",
        options=list(mcq["options"].keys()),
        format_func=lambda x: f"{x}) {mcq['options'][x]}",
        key=f"q_{st.session_state.session_vars['total_questions']}"
    )

    # Submit button with loading state
    if st.button("Submit Answer", key="submit"):
        with st.spinner("Checking answer..."):
            process_answer(selected_answer)

def process_answer(selected_answer):
    mcq = st.session_state.session_vars["current_mcq"]
    response_time = time.time() - st.session_state.session_vars["start_time"]
    
    # Update session state
    st.session_state.session_vars["total_questions"] += 1
    st.session_state.session_vars["question_times"].append(response_time)
    
    # Calculate points based on time and difficulty
    base_points = 10
    time_bonus = max(5 - int(response_time / 2), 0)
    difficulty_bonus = mcq["difficulty_level"] * 2
    points = base_points + time_bonus + difficulty_bonus
    
    if selected_answer == mcq["correct_answer"]:
        st.session_state.session_vars["score"] += points
        st.session_state.session_vars["correct_answers"] += 1
        st.success(f"✅ Correct! +{points} points")
    else:
        st.error(f"❌ Incorrect. The correct answer was {mcq['correct_answer']}) {mcq['options'][mcq['correct_answer']]}")
        st.session_state.session_vars["weak_topics"].extend(mcq["topics"])
    
    # Update performance data
    st.session_state.session_vars["performance_data"].append({
        "accuracy": (st.session_state.session_vars["correct_answers"] / 
                    st.session_state.session_vars["total_questions"] * 100),
        "response_time": response_time,
        "points": points if selected_answer == mcq["correct_answer"] else 0
    })
    
    st.session_state.session_vars["current_mcq"] = None

def reset_quiz():
    st.session_state.session_vars.update({
        "score": 0,
        "correct_answers": 0,
        "total_questions": 0,
        "current_mcq": None,
        "start_time": time.time(),
        "quiz_completed": False,
        "question_times": [],
        "question_history": [],
        "weak_topics": [],
        "performance_data": []
    })

def main():
    st.set_page_config(page_title="Smart MCQ Quiz", page_icon="📚", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #4CAF50;
            color: white;
        }
        .stProgress > div > div > div {
            background-color: #4CAF50;
        }
        .quiz-header {
            text-align: center;
            padding: 1em;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin-bottom: 2em;
        }
        .st-emotion-cache-16idsys p {
            font-size: 1.2em;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()

    # Header
    st.markdown("""
        <div class='quiz-header'>
            <h1>📚 Smart MCQ Quiz Generator</h1>
            <p>Upload a PDF and test your knowledge with automatically generated questions!</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Quiz Settings")
        difficulty = st.select_slider(
            "Question Difficulty",
            options=["Easy", "Medium", "Hard"],
            value="Medium"
        )
        
        max_questions = st.number_input(
            "Maximum Questions",
            min_value=5,
            max_value=20,
            value=5,
            step=5
        )
        
        if st.session_state.session_vars["quiz_started"]:
            if st.button("🔄 Reset Quiz"):
                reset_quiz()
                st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload section
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            process_uploaded_file(uploaded_file)
    
    with col2:
        if st.session_state.session_vars["chunks"]:
            st.success("PDF processed successfully! Ready to start quiz.")
            if not st.session_state.session_vars["quiz_started"]:
                if st.button("🎯 Start Quiz"):
                    start_quiz(max_questions)
                    st.rerun()

    # Quiz interface
    if st.session_state.session_vars["quiz_started"]:
        display_quiz_interface()

def process_uploaded_file(uploaded_file):
    """Process the uploaded PDF file with error handling and user feedback."""
    try:
        with st.spinner("Processing PDF..."):
            response = upload_pdf(uploaded_file)
            if response and "chunks" in response:
                st.session_state.session_vars["chunks"] = response["chunks"]
                return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

def start_quiz(max_questions=5):
    """Initialize a new quiz session."""
    st.session_state.session_vars.update({
        "quiz_started": True,
        "score": 0,
        "correct_answers": 0,
        "total_questions": 0,
        "current_mcq": None,
        "start_time": time.time(),
        "current_chunk_index": 0,
        "quiz_completed": False,
        "question_times": [],
        "question_history": [],
        "weak_topics": [],
        "max_questions": max_questions,
        "performance_data": []
    })

def fetch_mcq(context: str, difficulty: int = 2) -> Optional[Dict]:
    """Fetch MCQ with improved error handling and retry logic."""
    try:
        with st.spinner("Generating question... Please wait."):
            response = requests.post(
                f"{BASE_URL}/mcq/generate",
                json={"context": context, "difficulty": difficulty},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"MCQ API error: {response.status_code}")
                return None
                
    except requests.exceptions.Timeout:
        st.warning("Question generation is taking longer than expected. Retrying...")
        time.sleep(2)
        return fetch_mcq(context, difficulty)
    except Exception as e:
        logging.error(f"MCQ generation error: {str(e)}")
        return None

def upload_pdf(file) -> Optional[Dict]:
    """Upload and process PDF with improved error handling."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            f"{BASE_URL}/document/upload",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error processing PDF: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("PDF processing timed out. Please try again with a smaller file.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

if __name__ == "__main__":
    main()