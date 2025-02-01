import streamlit as st
import time
from mcq_generator import MCQGenerator
from document_processor import DocumentProcessor
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import numpy as np
from nlp_singleton import get_nlp

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
    """Generate an enhanced final report using vector embeddings and topic analysis."""
    try:
        session_vars = st.session_state.session_vars
        
        # Basic metrics
        total_questions = session_vars["total_questions"]
        correct_answers = session_vars["correct_answers"]
        accuracy = (correct_answers / max(total_questions, 1)) * 100
        total_time = sum(session_vars["question_times"])
        avg_time_per_question = total_time / max(total_questions, 1)
        
        # Get topic and difficulty analysis
        mcq_generator = session_vars["mcq_generator"]
        topic_analyzer = mcq_generator.topic_analyzer
        vector_store = llm["vector_store"]
        
        # Analyze wrong answers using embeddings
        wrong_answer_embeddings = []
        wrong_topics = []
        for i, attempted in enumerate(session_vars["questions_attempted"]):
            if attempted and i < len(session_vars["feedback"]):
                feedback = session_vars["feedback"][i]
                if not feedback.get("correct", False):
                    wrong_answer_embeddings.append(feedback.get("answer_embedding"))
                    wrong_topics.extend(feedback.get("topics", []))
        
        # Cluster wrong topics based on embeddings
        if wrong_answer_embeddings:
            similar_topics = find_similar_topics(wrong_answer_embeddings, wrong_topics, vector_store)
            topic_recommendations = generate_topic_recommendations(similar_topics, vector_store)
        else:
            similar_topics = []
            topic_recommendations = []
        
        # Generate performance insights
        performance_data = {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "avg_time": avg_time_per_question,
            "difficulty_stats": session_vars["difficulty_stats"],
            "topic_clusters": similar_topics,
            "recommendations": topic_recommendations
        }
        
        analysis_prompt = f"""
        Analyze quiz performance with these metrics:
        - Overall Score: {accuracy:.1f}%
        - Questions Answered: {total_questions}
        - Average Time: {avg_time_per_question:.1f} seconds
        - Difficulty Distribution: {session_vars["difficulty_stats"]}
        
        Topic Analysis:
        - Challenging Areas: {', '.join(similar_topics) if similar_topics else 'None identified'}
        
        Provide:
        1. Performance assessment
        2. Specific improvement recommendations for each weak topic
        3. Study strategy suggestions based on the difficulty distribution
        Keep the response actionable and specific.
        """
        
        # Display enhanced report
        st.markdown("### 📊 Detailed Performance Analysis")
        
        # Performance metrics
        cols = st.columns(4)
        cols[0].metric("Final Score", f"{accuracy:.1f}%")
        cols[1].metric("Questions Completed", total_questions)
        cols[2].metric("Avg. Response Time", f"{avg_time_per_question:.1f}s")
        cols[3].metric("Total Time", f"{total_time:.1f}s")
        
        # Difficulty breakdown
        st.markdown("### 📈 Question Difficulty Distribution")
        diff_cols = st.columns(3)
        for i, (diff, count) in enumerate(session_vars["difficulty_stats"].items()):
            diff_cols[i].metric(diff.capitalize(), count)
        
        # Topic analysis
        if similar_topics:
            st.markdown("### 📚 Topic Analysis")
            for topic_cluster in similar_topics:
                with st.expander(f"Topic Cluster: {topic_cluster['main_topic']}"):
                    st.write("Related Concepts:", ", ".join(topic_cluster['related_topics']))
                    st.write("Recommended Focus:", topic_cluster['recommendation'])
        
        # Generate AI analysis
        with st.spinner("Generating detailed analysis..."):
            try:
                analysis = llm["model"].invoke(analysis_prompt)
                if analysis and isinstance(analysis, str):
                    st.markdown("### 🤖 AI Performance Analysis")
                    st.markdown(analysis)
            except Exception as e:
                st.error(f"Could not generate AI analysis: {str(e)}")
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.markdown(f"""
        ### Basic Performance Summary
        Score: {session_vars.get('score', 0)}
        Questions Attempted: {session_vars.get('total_questions', 0)}
        """)

def find_similar_topics(wrong_answer_embeddings, wrong_topics, vector_store):
    """Cluster similar topics based on embedding similarity."""
    if not wrong_answer_embeddings:
        return []
    
    topic_clusters = []
    embeddings_array = np.array(wrong_answer_embeddings)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings_array)
    
    # Cluster similar topics
    processed_indices = set()
    for i in range(len(wrong_topics)):
        if i in processed_indices:
            continue
            
        cluster = {
            'main_topic': wrong_topics[i],
            'related_topics': set(),
            'recommendation': ''
        }
        
        # Find related topics based on embedding similarity
        for j in range(i + 1, len(wrong_topics)):
            if similarity_matrix[i][j] > 0.7:
                cluster['related_topics'].add(wrong_topics[j])
                processed_indices.add(j)
        
        topic_clusters.append(cluster)
        processed_indices.add(i)
    
    return topic_clusters

def cluster_topics(embeddings: List[np.ndarray], topics: List[str]) -> List[str]:
    """Cluster similar topics based on their vector embeddings."""
    if not embeddings or len(embeddings) < 2:
        return topics

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    clusters = []
    visited = set()

    for i in range(len(topics)):
        if i in visited:
            continue
        cluster = [topics[i]]
        visited.add(i)
        for j in range(i + 1, len(topics)):
            if j in visited:
                continue
            if similarity_matrix[i][j] > 0.7:  # Adjust threshold as needed
                cluster.append(topics[j])
                visited.add(j)
        clusters.append(", ".join(cluster))

    return clusters

def generate_topic_recommendations(topic_clusters, vector_store):
    """Generate specific recommendations for each topic cluster."""
    nlp = get_nlp()  # Get NLP instance
    recommendations = []
    
    for cluster in topic_clusters:
        # Find relevant content from vector store
        similar_content = vector_store.similarity_search(
            cluster['main_topic'],
            k=3
        )
        
        # Extract key concepts and generate recommendation
        concepts = set()
        for content in similar_content:
            doc = nlp(content.page_content)  # Use initialized NLP
            for ent in doc.ents:
                if ent.label_ in ['CONCEPT', 'TERM', 'TOPIC']:
                    concepts.add(ent.text)
        
        recommendation = {
            'topic': cluster['main_topic'],
            'related_concepts': list(concepts),
            'study_materials': [content.metadata.get('source', '') for content in similar_content]
        }
        recommendations.append(recommendation)
    
    return recommendations

        
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
        # Record wrong topics for future focus
        session_vars["mcq_generator"].record_wrong_answer(mcq)
    
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