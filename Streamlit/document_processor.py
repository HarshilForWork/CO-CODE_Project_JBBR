import os
import tempfile
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True
        )
    
    def process_pdf(self, file_content: bytes) -> List:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name
            
        try:
            loader = PDFPlumberLoader(temp_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    pass

    @staticmethod
    def init_llm():
        return {
            "embeddings": OllamaEmbeddings(model="deepseek-r1:8b"),
            "model": OllamaLLM(model="qwen2.5:7b"),
            "vector_store": InMemoryVectorStore(OllamaEmbeddings(model="deepseek-r1:8b"))
        }