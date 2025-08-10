# Advanced RAG Chatbot with Document Analysis

### **Live Demo:** [View the Deployed Application Here](https://document-chatbot-kambaa.vercel.app/)

An intelligent, full-stack chatbot application that leverages Retrieval-Augmented Generation (RAG) to answer questions based on user-uploaded documents. Built with a Python/FastAPI backend and a polished vanilla JavaScript frontend, this project is optimized for performance, user experience, and is fully deployed online.

## Key Features

*   **Multi-Format Document Upload:** Supports a wide range of document types (`.pdf`, `.docx`, `.txt`, and `.xlsx` Excel sheets) for building a dynamic knowledge base.
*   **Advanced RAG Architecture:** Utilizes `LangChain`, `FAISS` vector stores, and `HuggingFace` sentence transformers to provide accurate, context-aware answers strictly from the provided documents.
*   **Polished, Modern Frontend:**
    *   **Light & Dark Mode:** A sleek theme toggle that persists the user's choice in `localStorage`.
    *   **Responsive Design:** A clean, mobile-first interface that works beautifully on all screen sizes.
    *   **Accessibility Focused:** Includes support for `prefers-reduced-motion`, high-contrast modes, and clear `focus-visible` states.
    *   **Interactive UI:** Features smooth animations, drag-and-drop file uploads, and a "New Chat" functionality.
*   **Persistent User Sessions:** Chat history and uploaded documents are preserved on page reload within a single browser tab using `sessionStorage`. New tabs correctly initiate new sessions.
*   **Intelligent "Meta-Command" Handling:** The chatbot can answer questions about the application's state (e.g., "What documents have I uploaded?" or "What was my last question?") without confusing the RAG model.
*   **Optimized for Performance:** The heavy embeddings model is pre-loaded on application startup, not on each request, significantly reducing latency during file uploads.
*   **Cloud Deployed:** The entire application (backend API and frontend) is deployed on Render, demonstrating an understanding of modern CI/CD and cloud infrastructure practices.

## Tech Stack

| Area      | Technology / Library                                                              | Purpose                                                              |
| :-------- | :-------------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| **Backend** | **Python**, **FastAPI**, **Uvicorn**, **Gunicorn**                                | Building and serving the high-performance, asynchronous API.         |
|           | **LangChain**, **Groq**, **FAISS**, **HuggingFace**                               | Orchestrating the RAG pipeline for document analysis and generation. |
|           | **Pandas**, **PyPDF**, **python-docx**                                            | To parse and extract text from various document formats.             |
| **Frontend**| **HTML5**, **CSS3**, **Vanilla JavaScript (ES6)**                                   | Building a modern, accessible, and responsive user interface.        |
| **Deployment**| **Render**                                                                        | Cloud platform for deploying the backend and frontend.               |
|           | **Git & GitHub**                                                                  | For version control and as the deployment source for Render.         |

## Architectural Highlights & Problem-Solving

This project involved several key architectural decisions to ensure robustness, performance, and a superior user experience.

### 1. Performance Optimization: Solving the Embeddings Latency
**Problem:** The initial implementation loaded the `HuggingFaceEmbeddings` model on every file upload, causing a significant delay for the user.
**Solution:** I moved the model initialization to the global scope of the FastAPI application. This ensures the model is loaded only **once** when the server starts, making all subsequent file processing requests extremely fast.

### 2. Accuracy: Intelligent "Meta-Command" Interception
**Problem:** The RAG chain would fail when asked questions about the conversation itself (e.g., "What was my last question?"), as this information wasn't in the uploaded documents.
**Solution:** I implemented a command interception layer in the `/chat` endpoint. Using regular expressions, the backend now identifies these "meta-questions" and answers them using simple application logic, ensuring fast, accurate responses and preventing unnecessary RAG pipeline executions.

### 3. User-Centric Frontend with Modern Features
**Problem:** A standard UI is functional but not engaging. The goal was to create a polished and accessible experience.
**Solution:**
*   **Theme Engine:** I built a light/dark mode system using CSS variables (`:root`) and a `[data-theme]` attribute on the `<html>` element. This allows for instant theme changes without a page reload.
*   **State Persistence:** The user's theme preference is saved to `localStorage`, ensuring their choice is remembered on their next visit.
*   **Accessibility:** To ensure the application is usable by everyone, I included CSS media queries to respect user preferences for `prefers-reduced-motion` and `prefers-contrast`, along with clear focus indicators.

## Running the Project Locally

Follow these steps to set up and run the project on your local machine.

### Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/keys)

### 1. Clone the Repository
```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/[YOUR_REPO_NAME].git
cd [YOUR_REPO_NAME]
```

### 2. Backend Setup
```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt

# Create a .env file and add your Groq API key
echo "GROQ_API_KEY=your_actual_api_key_here" > .env

# Run the backend server
uvicorn main:app --reload
```
The backend will be running at `http://127.0.0.1:8000`.

### 3. Frontend Setup
Navigate to the `frontend` directory and open the `index.html` file in your web browser. It will connect to the local backend automatically.
