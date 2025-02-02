import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

# Initialize components
embeddings = SentenceTransformerEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:7b", temperature=0.7)

# Templates
topic_extraction_template = """
Analyze the following text and identify the main topic or concept being discussed.
Return only the topic name without any additional explanation.

Text: {text}

Topic:"""

flashcard_template = """
Create a flashcard about this important topic from the text. The flashcard should test understanding of key concepts.

Topic: {topic}
Context: {context}

Generate a flashcard in this format:
Q: [Question that tests understanding of this important topic]
A: [Comprehensive but concise answer]

Make the question challenging and focused on testing deep understanding rather than simple recall.
"""

# File handling setup
pdfs_directory = 'C:/PF/Projects/CO-CODE/Pdf_folder/'

def upload_pdf(file):
    """Save uploaded PDF file."""
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    """Load PDF file using PDFPlumber."""
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def extract_important_topics(documents, num_topics=10):
    """Extract important topics from documents using TF-IDF and LLM refinement."""
    # Combine all document chunks
    full_text = " ".join([doc.page_content for doc in documents])
    sentences = sent_tokenize(full_text)
    
    # Create and apply TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform([full_text])
    
    # Get and sort features by importance
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    important_phrases = sorted(
        zip(feature_names, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Extract and refine topics
    potential_topics = [phrase for phrase, score in important_phrases[:num_topics*2]]
    refined_topics = []
    
    for phrase in potential_topics:
        relevant_sentences = [s for s in sentences if phrase in s.lower()]
        if relevant_sentences:
            context = " ".join(relevant_sentences[:2])
            prompt = ChatPromptTemplate.from_template(topic_extraction_template)
            chain = prompt | model
            topic = chain.invoke({"text": context}).strip()
            
            if topic and topic not in refined_topics:
                refined_topics.append(topic)
                if len(refined_topics) == num_topics:
                    break
    
    return refined_topics

def generate_flashcards(documents, num_cards):
    """Generate flashcards for important topics."""
    topics = extract_important_topics(documents, num_topics=num_cards)
    prompt = ChatPromptTemplate.from_template(flashcard_template)
    chain = prompt | model
    flashcards = []
    
    for topic in topics:
        # Find relevant context
        relevant_docs = vector_store.similarity_search(topic, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate flashcard
        response = chain.invoke({
            "topic": topic,
            "context": context
        })
        
        # Parse response
        lines = response.split('\n')
        question = ""
        answer = ""
        for line in lines:
            if line.startswith('Q:'):
                question = line[2:].strip()
            elif line.startswith('A:'):
                answer = line[2:].strip()
        
        if question and answer:
            flashcards.append({
                "topic": topic,
                "question": question,
                "answer": answer
            })
    
    return flashcards

def generate_topic_specific_flashcards(documents, topic, num_cards=3):
    """Generate flashcards for a specific topic."""
    # Find relevant context for the topic
    relevant_docs = vector_store.similarity_search(topic, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Modified flashcard template for specific topics
    specific_flashcard_template = """
    Create {num_cards} different flashcards about this specific topic from the text. 
    Each flashcard should test different aspects of understanding.

    Topic: {topic}
    Context: {context}

    Generate flashcards in this format:
    Q1: [First question about the topic]
    A1: [Answer to first question]
    Q2: [Second question about the topic]
    A2: [Answer to second question]
    Q3: [Third question about the topic]
    A3: [Answer to third question]

    Make questions challenging and focused on testing deep understanding rather than simple recall.
    """
    
    prompt = ChatPromptTemplate.from_template(specific_flashcard_template)
    chain = prompt | model
    
    # Generate flashcards
    response = chain.invoke({
        "topic": topic,
        "context": context,
        "num_cards": num_cards
    })
    
    # Parse response into flashcards
    flashcards = []
    lines = response.split('\n')
    current_q = ""
    current_a = ""
    
    for line in lines:
        if line.startswith('Q'):
            if current_q and current_a:
                flashcards.append({
                    "topic": topic,
                    "question": current_q,
                    "answer": current_a
                })
            current_q = line[3:].strip()
        elif line.startswith('A'):
            current_a = line[3:].strip()
    
    # Add the last flashcard
    if current_q and current_a:
        flashcards.append({
            "topic": topic,
            "question": current_q,
            "answer": current_a
        })
    
    return flashcards

# Initialize session states
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = []
if 'current_card_index' not in st.session_state:
    st.session_state.current_card_index = 0
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'generation_mode' not in st.session_state:
    st.session_state.generation_mode = 'automatic'
if 'custom_topic' not in st.session_state:
    st.session_state.custom_topic = ''

# Callback functions
def next_card():
    """Move to next flashcard."""
    if st.session_state.current_card_index < len(st.session_state.flashcards) - 1:
        st.session_state.current_card_index += 1
        st.session_state.show_answer = False

def prev_card():
    """Move to previous flashcard."""
    if st.session_state.current_card_index > 0:
        st.session_state.current_card_index -= 1
        st.session_state.show_answer = False

def toggle_answer():
    """Toggle answer visibility."""
    st.session_state.show_answer = not st.session_state.show_answer

def generate_cards():
    """Generate flashcards based on selected mode."""
    with st.spinner("Analyzing document and generating flashcards..."):
        if st.session_state.generation_mode == 'automatic':
            st.session_state.flashcards = generate_flashcards(
                st.session_state.chunked_documents,
                st.session_state.num_cards
            )
        else:  # specific topic mode
            st.session_state.flashcards = generate_topic_specific_flashcards(
                st.session_state.chunked_documents,
                st.session_state.custom_topic,
                st.session_state.num_cards
            )
        st.session_state.current_card_index = 0
        st.session_state.show_answer = False
        st.session_state.processing_complete = True

def reset_state():
    """Reset application state."""
    st.session_state.flashcards = []
    st.session_state.current_card_index = 0
    st.session_state.show_answer = False
    st.session_state.processing_complete = False
    st.session_state.pdf_processed = False
    st.session_state.generation_mode = 'automatic'
    st.session_state.custom_topic = ''

# Streamlit UI
st.title("PDF Flashcard Generator")
st.write("This system automatically generates flashcards for the most important topics in your PDF.")

# File upload section
uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False, on_change=reset_state)

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF..."):
        upload_pdf(uploaded_file)
        documents = load_pdf(pdfs_directory + uploaded_file.name)
        st.session_state.chunked_documents = split_text(documents)
        vector_store.add_documents(st.session_state.chunked_documents)
        st.session_state.pdf_processed = True
        st.success("PDF processed successfully!")

# Flashcard generation options
if st.session_state.pdf_processed and not st.session_state.processing_complete:
    st.session_state.generation_mode = st.radio(
        "Flashcard Generation Mode",
        ['automatic', 'specific topic'],
        help="Choose whether to automatically identify important topics or focus on a specific topic"
    )
    
    if st.session_state.generation_mode == 'automatic':
        st.session_state.num_cards = st.number_input(
            "Number of important topics to generate flashcards for",
            min_value=1,
            max_value=10,
            value=5,
            help="The system will identify this many important topics and create flashcards for them."
        )
    else:
        st.session_state.custom_topic = st.text_input(
            "Enter the specific topic you want to study",
            help="The system will generate flashcards focused on this topic"
        )
        st.session_state.num_cards = st.number_input(
            "Number of flashcards to generate for this topic",
            min_value=1,
            max_value=5,
            value=3,
            help="The system will generate this many flashcards for your chosen topic."
        )
    
    generate_button = st.button(
        "Generate Flashcards",
        on_click=generate_cards,
        type="primary",
        disabled=st.session_state.generation_mode == 'specific topic' and not st.session_state.custom_topic
    )

# Display flashcards
if st.session_state.flashcards:
    st.subheader(f"Flashcard {st.session_state.current_card_index + 1} of {len(st.session_state.flashcards)}")
    
    current_card = st.session_state.flashcards[st.session_state.current_card_index]
    
    # Display card content
    with st.container():
        st.markdown("**Topic:**")
        st.info(current_card["topic"])
        
        st.markdown("**Question:**")
        st.write(current_card["question"])
        
        # Toggle answer button
        st.button(
            "Show Answer" if not st.session_state.show_answer else "Hide Answer",
            on_click=toggle_answer,
            key="toggle_answer"
        )
        
        if st.session_state.show_answer:
            st.markdown("**Answer:**")
            st.success(current_card["answer"])
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "Previous",
            on_click=prev_card,
            disabled=st.session_state.current_card_index == 0
        )
    with col2:
        st.button(
            "Next",
            on_click=next_card,
            disabled=st.session_state.current_card_index == len(st.session_state.flashcards) - 1
        )
    
    # Progress bar
    progress = (st.session_state.current_card_index + 1) / len(st.session_state.flashcards)
    st.progress(progress)
    
    # Study progress
    st.caption(f"Progress: {st.session_state.current_card_index + 1} of {len(st.session_state.flashcards)} cards")