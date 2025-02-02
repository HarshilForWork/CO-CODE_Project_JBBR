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
model = OllamaLLM(model="qwen2.5:7b",temperature=0.7)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
Question: {question} 
Context: {context}
Answer:
"""

pdfs_directory = 'C:/PF/Projects/CO-CODE/Pdf_folder/'

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

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

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

    question = st.chat_input("Ask a question about your PDF")

    if question and st.session_state.vector_store_initialized:
        with st.spinner("Finding answer..."):
            st.chat_message("user").write(question)
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)
            st.chat_message("assistant").write(answer)
