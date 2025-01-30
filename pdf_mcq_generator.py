import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define the prompt template for generating MCQs
mcq_template = """
You are an assistant for generating multiple-choice questions (MCQs) based on the provided context. 
Generate one MCQ with four options (A, B, C, D), where only one option is correct. 
Include the correct answer and assign a difficulty level between 1 (easy) and 4 (hard).
Context: {context}
Output format:
Question: [Your question here]
Options:
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [Correct option, e.g., A]
Difficulty Level: [1-4]
"""

# Initialize Ollama embeddings and models
embeddings = OllamaEmbeddings(model="qwen2:7b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:8b")

# Function to load PDF file
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Function to index documents into the vector store
def index_docs(documents):
    vector_store.add_documents(documents)

# Function to retrieve relevant documents based on a query
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Function to generate an MCQ
def generate_mcq(context):
    prompt = ChatPromptTemplate.from_template(mcq_template)
    chain = prompt | model
    response = chain.invoke({"context": context})
    return response

# Create a temporary directory if it doesn't exist
temp_dir = tempfile.gettempdir()

# Streamlit UI for uploading PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    # Create a temporary file path using the system's temp directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the PDF
        documents = load_pdf(file_path)
        chunked_documents = split_text(documents)
        index_docs(chunked_documents)
        
        # Generate MCQ button
        if st.button("Generate MCQ"):
            random_chunk = chunked_documents[0].page_content  # Using the first chunk
            mcq_response = generate_mcq(random_chunk)
            
            # Parse the MCQ response
            lines = mcq_response.strip().split("\n")
            question = lines[0].replace("Question: ", "")
            options = {
                "A": lines[1].replace("A) ", ""),
                "B": lines[2].replace("B) ", ""),
                "C": lines[3].replace("C) ", ""),
                "D": lines[4].replace("D) ", "")
            }
            correct_answer = lines[5].replace("Correct Answer: ", "")
            difficulty_level = int(lines[6].replace("Difficulty Level: ", ""))
            
            # Display the MCQ
            st.subheader("Generated MCQ:")
            st.write(f"**Question:** {question}")
            st.write(f"A) {options['A']}")
            st.write(f"B) {options['B']}")
            st.write(f"C) {options['C']}")
            st.write(f"D) {options['D']}")
            st.write(f"**Difficulty Level:** {difficulty_level}")
            
            # User selects an answer
            user_answer = st.radio("Select your answer:", list(options.keys()))
            if st.button("Submit Answer"):
                if user_answer == correct_answer:
                    st.success("Correct!")
                else:
                    st.error(f"Wrong! The correct answer is {correct_answer}.")
    
    finally:
        # Clean up: remove the temporary file after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Error cleaning up temporary file: {e}")