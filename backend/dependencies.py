from Streamlit.mcq_generator import OptimizedMCQGenerator
from Streamlit.document_processor import DocumentProcessor
from Streamlit.nlp_singleton import get_nlp

# Singleton instances
_mcq_generator = None
_document_processor = None
_llm_components = None

def get_llm_components():
    global _llm_components
    if _llm_components is None:
        _llm_components = DocumentProcessor.init_llm()
    return _llm_components

def get_mcq_generator():
    global _mcq_generator
    if _mcq_generator is None:
        llm = get_llm_components()
        _mcq_generator = OptimizedMCQGenerator(llm)
    return _mcq_generator

def get_document_processor():
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor