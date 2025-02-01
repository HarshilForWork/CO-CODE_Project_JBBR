# PDF MCQ Generator using Streamlit and Ollama

This project is a **PDF-based MCQ Generator** built with **Streamlit** and **LangChain**, utilizing **Qwen2** for reading PDFs and **DeepSeek** for generating questions.

## Features

- Upload a **PDF from anywhere** and extract text using Qwen2.
- Split the extracted text into **manageable chunks** for processing.
- Generate **multiple-choice questions (MCQs)** based on extracted content using DeepSeek.
- Display **difficulty levels** and allow users to select answers interactively.

## Tech Stack

- **Streamlit** - Web interface
- **LangChain** - Text processing and retrieval
- **Ollama** - Local AI models (Qwen2 & DeepSeek)
- **PDFPlumber** - PDF text extraction

#### Note: Additionally, you may need to run:

run the requirements.txt

**python -m spacy download en_core_web_sm
nltk.download('wordnet')**
