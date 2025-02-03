# document_processor.py
import os
import tempfile
import logging
from typing import List, Dict
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up global cache
set_llm_cache(InMemoryCache())

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True
        )
    
    @staticmethod
    def verify_ollama_model(model_name: str) -> bool:
        """Verify if an Ollama model is available."""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json()["models"]]
                return model_name in available_models
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama model {model_name}: {str(e)}")
            return False

    @staticmethod
    def init_llm() -> Dict:
        """Initialize the LLM and embeddings with specific models for different tasks."""
        logger.info("Initializing LLM components...")
        
        # Check if Ollama is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama server is not running")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            raise Exception("Ollama server is not accessible. Please ensure Ollama is installed and running.")

        try:
            # Initialize embeddings with fallback options
            embedding_models = ["deepseek-r1:8b"]
            embeddings = None
            
            for model in embedding_models:
                if DocumentProcessor.verify_ollama_model(model):
                    logger.info(f"Using {model} for embeddings")
                    embeddings = OllamaEmbeddings(
                        model=model,
                        temperature=0.1,
                    )
                    break
            
            if not embeddings:
                raise Exception("No suitable embedding model found")

            # Initialize main LLM with fallback options
            llm_models = ["qwen2.5:7b"]
            main_llm = None
            
            for model in llm_models:
                if DocumentProcessor.verify_ollama_model(model):
                    logger.info(f"Using {model} for main LLM")
                    main_llm = OllamaLLM(
                        model=model,
                        temperature=0.1,
                        num_ctx=2048,
                    )
                    break
            
            if not main_llm:
                raise Exception("No suitable LLM model found")

            # Initialize report LLM
            report_models = ["qwen2.5:3b"]
            report_llm = None
            
            for model in report_models:
                if DocumentProcessor.verify_ollama_model(model):
                    logger.info(f"Using {model} for report generation")
                    report_llm = OllamaLLM(
                        model=model,
                        temperature=0.1,
                        num_ctx=2048,
                    )
                    break
            
            if not report_llm:
                raise Exception("No suitable report model found")

            return {
                "embeddings": embeddings,
                "model": main_llm,
                "report_model": report_llm,
                "vector_store": InMemoryVectorStore(embeddings)
            }

        except Exception as e:
            logger.error(f"Failed to initialize LLM components: {str(e)}")
            raise Exception(f"LLM initialization failed: {str(e)}")
        
    def process_pdf(self, file_content: bytes) -> List:
        """Process a PDF file and split it into chunks."""
        # Create a temporary file with proper cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            try:
                logger.info("Creating temporary PDF file...")
                tmp_file.write(file_content)
                tmp_file.flush()  # Ensure all data is written
                temp_path = tmp_file.name
                
                logger.info("Loading PDF with PDFPlumberLoader...")
                loader = PDFPlumberLoader(temp_path)
                documents = loader.load()
                
                if not documents:
                    logger.error("No text found in the PDF.")
                    raise ValueError("No text found in the PDF.")
                
                logger.info(f"PDF loaded successfully. Found {len(documents)} pages.")
                chunks = self.text_splitter.split_documents(documents)
                logger.info(f"Split into {len(chunks)} chunks.")
                
                return chunks
                
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                raise ValueError(f"Error processing PDF: {str(e)}")
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except (PermissionError, FileNotFoundError, UnboundLocalError):
                    pass