import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from pathlib import Path

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

# Download NLTK data with specific path
nltk.data.path.append("C:\\Users\\Jay Manish Guri\\AppData\\Roaming\\nltk_data")
nltk.download('punkt')
nltk.download('stopwords', quiet=False)

# Initialize FastAPI app
app = FastAPI(title="Flashcard Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage for flashcards (can be replaced with Redis for persistence)
flashcard_sessions = {}

# Pydantic models
class FlashCard(BaseModel):
    topic: str
    question: str
    answer: str

class FlashCardResponse(BaseModel):
    flashcards: List[FlashCard]

class TopicRequest(BaseModel):
    topic: str
    num_cards: int = 3

class NextFlashcardRequest(BaseModel):
    session_id: str

class FlashCardSessionResponse(BaseModel):
    session_id: str
    flashcard: Optional[FlashCard]

# Initialize Components
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Global components
embeddings = SentenceTransformerEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:7b", temperature=0.7)

# Prompt Templates
topic_extraction_template = """
Analyze the following text and identify the main topic or concept being discussed.
Return only the topic name.

Text: {text}

Topic:"""

flashcard_template = """
Create a flashcard about this important topic from the text. The flashcard should test understanding of key concepts.

Topic: {topic}
Context: {context}

Generate a flashcard in this format:
Q: [Question that tests understanding]
A: [Comprehensive but concise answer]
"""

# Folder for storing uploaded PDFs
UPLOAD_FOLDER = "uploaded_pdfs"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

async def process_pdf(file_path: str):
    """Process the uploaded PDF file from the saved path."""
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(400, "No text extracted from PDF.")

        return documents

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(400, f"Error processing PDF: {str(e)}")

def extract_important_topics(documents, num_topics=10):
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
    
    potential_topics = [phrase for phrase, score in important_phrases[:num_topics*2]]
    refined_topics = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for phrase in potential_topics:
            relevant_sentences = [s for s in sentences if phrase in s.lower()]
            if relevant_sentences:
                context = " ".join(relevant_sentences[:2])
                prompt = ChatPromptTemplate.from_template(topic_extraction_template)
                chain = prompt | model
                futures.append(executor.submit(chain.invoke, {"text": context}))
        
        for future in futures:
            topic = future.result().strip()
            if topic and topic not in refined_topics:
                refined_topics.append(topic)
                if len(refined_topics) == num_topics:
                    break
    
    return refined_topics

async def generate_flashcards_for_topics(documents, topics):
    """Generate flashcards asynchronously."""
    flashcards = []
    prompt = ChatPromptTemplate.from_template(flashcard_template)
    chain = prompt | model
    
    def process_topic(topic):
        relevant_docs = vector_store.similarity_search(topic, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        response = chain.invoke({"topic": topic, "context": context})
        return parse_flashcard_response(response, topic)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_topic, topic) for topic in topics]
        for future in futures:
            flashcard = future.result()
            if flashcard:
                flashcards.append(flashcard)
    
    return flashcards

def parse_flashcard_response(response: str, topic: str) -> Optional[FlashCard]:
    """Parse LLM response into a structured flashcard."""
    lines = response.split('\n')
    question, answer = "", ""
    
    for line in lines:
        if line.startswith('Q:'):
            question = line[2:].strip()
        elif line.startswith('A:'):
            answer = line[2:].strip()
    
    return FlashCard(topic=topic, question=question, answer=answer) if question and answer else None

@app.get("/")
def read_root():
    return {"message": "CORS is enabled!"}

@app.post("/upload", response_model=FlashCardSessionResponse)
async def upload_pdf(file: UploadFile = File(...), num_cards: int = 5):
    """Upload PDF and generate flashcards."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    # Save the uploaded file to the server
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid4()}.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    documents = await process_pdf(file_path)
    
    topics = extract_important_topics(documents, num_topics=num_cards)
    flashcards = await generate_flashcards_for_topics(documents, topics)
    
    session_id = str(uuid4())
    flashcard_sessions[session_id] = flashcards
    
    return FlashCardSessionResponse(session_id=session_id, flashcard=flashcards[0] if flashcards else None)

@app.post("/next-flashcard", response_model=FlashCardSessionResponse)
async def get_next_flashcard(request: NextFlashcardRequest):
    """Fetch the next flashcard from the session."""
    session_id = request.session_id
    if session_id not in flashcard_sessions or not flashcard_sessions[session_id]:
        raise HTTPException(404, "No more flashcards available")
    
    return FlashCardSessionResponse(session_id=session_id, flashcard=flashcard_sessions[session_id].pop(0))

if __name__ == "__main__":
    uvicorn.run("main_flash:app", host="0.0.0.0", port=8000, reload=True)