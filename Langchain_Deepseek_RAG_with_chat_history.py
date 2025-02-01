import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import numpy as np

# Custom embedding class to wrap SentenceTransformer
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

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
Previous Conversations (User-Assistant Q&A): {previous_conversations}
Chat History (User-Assistant Dialogue): {chat_history}
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'C:/PF/Projects/CO-CODE/Pdf_folder/'

# Store conversation and chat history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, previous_conversations, chat_history, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    conversation_history = "\n".join(previous_conversations)  # Include previous Q&A
    chat_history_str = "\n".join(chat_history)  # Include user-assistant chat history
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "previous_conversations": conversation_history, "chat_history": chat_history_str, "context": context})

# Streamlit interface
st.title("PDF Question Answering System")

if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        upload_pdf(uploaded_file)
        documents = load_pdf(pdfs_directory + uploaded_file.name)
        chunked_documents = split_text(documents)
        index_docs(chunked_documents)
        st.session_state.vector_store_initialized = True
        st.success("PDF processed successfully!")

# Display the chat history before the user inputs a question
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        if "User:" in message:
            st.chat_message("user").write(message.replace("User:", "").strip())
        else:
            st.chat_message("assistant").write(message.replace("Assistant:", "").strip())

question = st.chat_input("Ask a question about your PDF")

if question and st.session_state.vector_store_initialized:
    with st.spinner("Finding answer..."):
        st.chat_message("user").write(question)
        
        # Retrieve related documents based on the current question
        related_documents = retrieve_docs(question)
        
        # Add the current question and previous conversation to the context
        answer = answer_question(question, st.session_state.conversation_history, st.session_state.chat_history, related_documents)
        st.chat_message("assistant").write(answer)

        # Add the current question and answer to the conversation history for context in the next turn
        st.session_state.conversation_history.append(f"User: {question}")
        st.session_state.conversation_history.append(f"Assistant: {answer}")

        # Add the current question and answer to the chat history
        st.session_state.chat_history.append(f"User: {question}")
        st.session_state.chat_history.append(f"Assistant: {answer}")
