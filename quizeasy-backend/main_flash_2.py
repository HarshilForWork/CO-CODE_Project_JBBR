import os
import json
from uuid import uuid4
from pathlib import Path
import uvicorn
import nltk
import numpy as np
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Natural Language Processing Libraries
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine Learning and NLP Libraries
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama.llms import OllamaLLM

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize FastAPI app
app = FastAPI(title="Flashcard Generator API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder for storing uploaded PDFs
UPLOAD_FOLDER = "uploaded_pdfs"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Session storage for flashcards
flashcard_sessions = {}

# Custom Embedding Class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Initialize Components
embeddings = SentenceTransformerEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:7b", temperature=0.7)

# Pydantic Models
class FlashCard(BaseModel):
    topic: str
    question: str
    answer: str

class FlashCardSessionResponse(BaseModel):
    session_id: str
    flashcard: Optional[FlashCard]
    remaining_cards: int
    total_cards: int

# Prompt Templates
topic_extraction_template = """
Analyze the following text and identify the most important topic or concept.

Text: {text}

Topic:"""

flashcard_template = """
Create a flashcard about this important topic from the text that tests understanding of key concepts.

Topic: {topic}
Context: {context}

Generate a flashcard in this format:
Q: [Question that tests understanding]
A: [Comprehensive and clear answer]
"""

# Helper Functions
async def process_pdf(file_path: str):
    """Process the uploaded PDF file."""
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(400, "No text extracted from PDF.")

        return documents

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(400, f"Error processing PDF: {str(e)}")

def extract_important_topics(documents, num_topics=10, specific_topics=None):
    """Extract important topics using TF-IDF and LLM refinement."""
    full_text = " ".join([doc.page_content for doc in documents])
    sentences = sent_tokenize(full_text)
    
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform([full_text])
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    important_phrases = sorted(
        zip(feature_names, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Prioritize specific topics if provided
    if specific_topics:
        matched_topics = [
            topic for topic in specific_topics 
            if any(topic.lower() in phrase.lower() for phrase, _ in important_phrases)
        ]
        
        if matched_topics:
            return matched_topics[:num_topics]
    
    # Fallback to TF-IDF approach
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

async def generate_flashcards_for_topics(documents, topics, specific_topics=None):
    """Generate flashcards asynchronously."""
    flashcards = []
    prompt = ChatPromptTemplate.from_template(flashcard_template)
    chain = prompt | model
    
    for topic in topics:
        relevant_docs = vector_store.similarity_search(topic, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        response = chain.invoke({
            "topic": topic, 
            "context": context
        })
        
        # Parse flashcard response
        lines = response.split('\n')
        question, answer = "", ""
        
        for line in lines:
            if line.startswith('Q:'):
                question = line[2:].strip()
            elif line.startswith('A:'):
                answer = line[2:].strip()
        
        if question and answer:
            flashcards.append(FlashCard(topic=topic, question=question, answer=answer))
        
        if len(flashcards) == len(topics):
            break
    
    return flashcards

# Main API Endpoint
@app.post("/upload", response_model=FlashCardSessionResponse)
async def upload_pdf(
    file: UploadFile = File(...), 
    num_cards: str = Form(default="5"),
    specific_topics: str = Form(default="")
):
    """Upload PDF and generate flashcards with frontend input."""
    # Convert input parameters
    try:
        num_cards = int(num_cards)
    except ValueError:
        num_cards = 5  # Default to 5 if invalid input
    
    # Parse specific topics
    if specific_topics:
        try:
            specific_topics = json.loads(specific_topics)
        except (json.JSONDecodeError, TypeError):
            specific_topics = [topic.strip() for topic in specific_topics.split(',') if topic.strip()]
    else:
        specific_topics = None
    
    # Validate number of cards
    num_cards = max(1, min(num_cards, 20))  # Clamp between 1 and 20
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    # Save the uploaded file to the server
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid4()}.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Process PDF and generate flashcards
        documents = await process_pdf(file_path)
        
        # Extract topics with optional specific topic filtering
        topics = extract_important_topics(
            documents, 
            num_topics=num_cards, 
            specific_topics=specific_topics
        )
        
        # Generate flashcards with specified parameters
        flashcards = await generate_flashcards_for_topics(
            documents, 
            topics, 
            specific_topics=specific_topics
        )
        
        # Trim to requested number of cards
        flashcards = flashcards[:num_cards]
        
        # Create session and return first flashcard
        session_id = str(uuid4())
        flashcard_sessions[session_id] = flashcards
        
        return FlashCardSessionResponse(
            session_id=session_id, 
            flashcard=flashcards[0] if flashcards else None,
            remaining_cards=len(flashcards) - 1,
            total_cards=len(flashcards)
        )
    
    finally:
        # Clean up the uploaded file
        os.remove(file_path)

# Next Flashcard Endpoint
@app.post("/next-flashcard")
async def get_next_flashcard(session_id: str = Form(...)):
    """Fetch the next flashcard from the session."""
    if session_id not in flashcard_sessions or not flashcard_sessions[session_id]:
        raise HTTPException(404, "No more flashcards available")
    
    remaining_flashcards = flashcard_sessions[session_id]
    next_flashcard = remaining_flashcards.pop(0)
    
    return {
        "session_id": session_id, 
        "flashcard": next_flashcard,
        "remaining_cards": len(remaining_flashcards),
        "total_cards": len(remaining_flashcards) + 1
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)