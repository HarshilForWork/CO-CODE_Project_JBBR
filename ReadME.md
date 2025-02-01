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

Run the requirements.txt:

```bash
pip install -r requirements.txt
```

Then, install additional dependencies:

```bash
python -m spacy download en_core_web_sm

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## **Workflow of the PDF-based MCQ Generation System**

This system processes a PDF document, extracts key information, generates multiple-choice questions (MCQs), and presents an interactive quiz interface. Below is the structured workflow detailing how each component interacts.

### **1. User Uploads PDF (Frontend - `app.py`)**

- The user uploads a PDF via Streamlit’s file uploader.
- The file’s content is passed to the **DocumentProcessor** for processing.

### **2. Document Processing (`document_processor.py`)**

- **Extract Text from PDF**
  - The PDF content is read using `PDFPlumberLoader`.
  - Extracted text is split into meaningful chunks using `RecursiveCharacterTextSplitter`.
- **Initialize LLM and Vector Store**
  - `OllamaLLM` is initialized for text generation.
  - `OllamaEmbeddings` is used for vector representation of text.
  - `InMemoryVectorStore` is created for efficient retrieval.
- **Return Processed Chunks**
  - Processed text chunks are returned to `app.py` for MCQ generation.

### **3. Generate MCQs (`mcq_generator.py`)**

- **Select a Text Chunk**
  - The next chunk of text is chosen for generating questions.
- **Generate Question using LLM**
  - A prompt is sent to `OllamaLLM` to generate a meaningful question from the chunk.
- **Extract Answer using BERT QA Pipeline**
  - A **question-answering model** (`distilbert-base-uncased-distilled-squad`) identifies the correct answer.
- **Generate Distractors**
  - **Synonym-based** (using `wordnet` from `nltk`).
  - **Context-based** (using `spacy` to extract similar words/noun phrases).
  - **Fallback methods** (if insufficient distractors are found, words are rearranged).
- **Assess Difficulty Level**
  - Difficulty is determined based on question complexity, answer length, and presence of named entities.
- **Return MCQ**
  - An MCQ object is created with:
    - **Question**
    - **Options (A, B, C, D)**
    - **Correct Answer**
    - **Difficulty Level**
  - The MCQ is sent back to `app.py`.

### **4. Interactive Quiz (Frontend - `app.py`)**

- **Display MCQ to User**
  - The question and four options are presented.
- **User Submits Answer**
  - The response is checked against the correct answer.
  - Score is updated based on correctness and response time.
- **Track Performance**
  - Correct and incorrect attempts are recorded.
  - Time taken per question is logged.
- **Move to Next Question**
  - If less than 5 questions have been answered, the next MCQ is generated.
- **Quiz Completion**
  - When all 5 questions are answered, a final report is generated.

### **5. Performance Analysis (AI Feedback - `app.py`)**

- **Calculate Quiz Metrics**
  - Total score
  - Accuracy percentage
  - Average response time
  - Correct vs. incorrect answers
- **Generate AI-based Feedback**
  - `OllamaLLM` provides:
    - Performance summary
    - Time management insights
    - Two specific improvement tips
- **Display Results to User**
  - The report is displayed in a structured format.

## **Component Responsibilities**

| **Component**                                      | **Responsibilities**                                                                   |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Frontend (`app.py`)**                          | Handles user interaction, displays MCQs, collects responses, and generates the final report. |
| **Document Processor (`document_processor.py`)** | Extracts text from PDFs, splits text into chunks, and initializes the LLM.                   |
| **MCQ Generator (`mcq_generator.py`)**           | Generates questions, answers, and distractors, assesses difficulty, and returns MCQs.        |
| **LLM & Embeddings**                               | Generates questions (OllamaLLM) and provides vector embeddings for text retrieval.           |
| **BERT QA Model**                                  | Extracts the correct answer from text.                                                       |
| **NLP Components (`spacy`, `nltk`)**           | Used for synonym generation and distractor selection.                                        |

---

This ensures a **dynamic question generation**, **interactive learning**, and **AI-driven feedback** for an engaging quiz experience. 🚀
