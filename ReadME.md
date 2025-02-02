
# PDF MCQ Generator using Streamlit and Ollama

This project is a **PDF-based MCQ Generator** built with **Streamlit** and  **LangChain** , utilizing **Qwen2** for reading PDFs and **DeepSeek** for generating multiple-choice questions (MCQs).

## Features 🚀

* **Upload a PDF** and extract text automatically.
* **Generate MCQs dynamically** using AI-powered models.
* **Track quiz performance** with real-time metrics.
* **Receive AI-based feedback** for improvement.

## Tech Stack 🛠️

* **Streamlit** - Web interface
* **LangChain** - Text processing and retrieval
* **Ollama** - Local AI models (Qwen2 & DeepSeek)
* **PDFPlumber** - PDF text extraction
* **spaCy & NLTK** - NLP-based distractor generation
* **DistilBERT** - Extract answers from context
* **FAISS** - Vector-based retrieval

## Installation 🔧

### **1. Clone the Repository**

```bash
git clone https://github.com/your-repo/pdf-mcq-generator.git
cd pdf-mcq-generator
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Download NLP Models**

```bash
python -m spacy download en_core_web_sm

import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

### **4. Run the Application**

```bash
streamlit run app.py
```

---

## **Workflow of the PDF-based MCQ Generation System**

This system processes a PDF, extracts key information, generates multiple-choice questions (MCQs), and presents an interactive quiz interface.

### **1. User Uploads PDF (`app.py`)**

* The user uploads a PDF via Streamlit’s file uploader.
* The file content is passed to `DocumentProcessor` for processing.

### **2. Document Processing (`document_processor.py`)**

* **Extract Text from PDF** → Uses `PDFPlumberLoader`.
* **Split Text into Chunks** → Uses `RecursiveCharacterTextSplitter`.
* **Initialize AI Models** → Uses `OllamaLLM` and `InMemoryVectorStore`.
* **Return Processed Chunks** to `app.py` for MCQ generation.

### **3. Generate MCQs (`mcq_generator.py`)**

* **Select a Text Chunk** → Choose a portion of text.
* **Generate Question using LLM** → Prompt `OllamaLLM` to create a question.
* **Extract Answer using BERT QA Pipeline** → Identify correct answers.
* **Generate Distractors** → Find incorrect answer choices.
* **Assess Difficulty Level** → Determine question complexity.
* **Return MCQ** → Send question and options back to the frontend.

### **4. Interactive Quiz (`app.py`)**

* **Display MCQ to User** → Show questions and answer choices.
* **User Submits Answer** → Validate the response.
* **Track Performance** → Store correct/incorrect attempts.
* **Move to Next Question** → Continue the quiz.
* **Quiz Completion** → Generate a final report.

### **5. AI-Based Performance Analysis (`app.py`)**

* **Calculate Quiz Metrics** → Score, accuracy, response time.
* **Generate AI-Based Feedback** → Insights and improvement tips.
* **Display Results to User** → Show a structured performance summary.

---

## **Component Responsibilities**

| **Component**                  | **Responsibilities**                             |
| ------------------------------------ | ------------------------------------------------------ |
| **Frontend (`app.py`)**      | Displays MCQs, collects responses, and shows results.  |
| **Document Processor**         | Extracts and splits text from PDFs.                    |
| **MCQ Generator**              | Generates MCQs with distractors and difficulty levels. |
| **AI Models (`OllamaLLM`)**  | Creates questions and provides AI-based feedback.      |
| **BERT QA Model**              | Extracts correct answers from the text.                |
| **NLP Components (`spaCy`)** | Generates synonyms and distractors.                    |

This system ensures **dynamic question generation, interactive learning, and AI-driven feedback** for an engaging quiz experience. 🚀
