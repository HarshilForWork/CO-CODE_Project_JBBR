# document_processor.py
import os
import tempfile
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Set up global cache
set_llm_cache(InMemoryCache())

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True
        )
    
    def process_pdf(self, file_content: bytes) -> List:
        """Process a PDF file and split it into chunks."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name
            
        try:
            loader = PDFPlumberLoader(temp_path)
            documents = loader.load()
            if not documents:
                raise ValueError("No text found in the PDF.")
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    pass

    @staticmethod
    def init_llm():
        """Initialize the LLM and embeddings."""
        embeddings = OllamaEmbeddings(
            model="deepseek-r1:8b",
            temperature=0.1,
        )
        
        llm = OllamaLLM(
            model="qwen2.5:7b",
            temperature=0.1,
            num_ctx=2048,
        )
        
        return {
            "embeddings": embeddings,
            "model": llm,
            "vector_store": InMemoryVectorStore(embeddings)
        }