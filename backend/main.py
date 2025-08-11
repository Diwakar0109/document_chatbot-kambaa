# File: backend/main.py

import os
import io
import re
from dotenv import load_dotenv
from typing import List, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Document Parsers ---
from pypdf import PdfReader
from docx import Document
import pandas as pd

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

# --- App Initialization ---
app = FastAPI(
    title="AI Chatbot Backend",
    description="Backend for a RAG chatbot with session management.",
    version="1.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://document-chatbot-kambaa.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pre-load Models on Startup for Efficiency ---
print("Loading HuggingFace embeddings model...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("Embeddings model loaded successfully.")
except Exception as e:
    print(f"Error loading embeddings model: {e}")
    embeddings = None


# --- Session Management ---
# In-memory dictionary for session state. Replace with Redis for production.
sessions: Dict[str, Dict] = {}


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

class SessionRequest(BaseModel):
    session_id: str


# --- Helper Functions ---
def extract_text_from_file(file: UploadFile, file_contents: bytes) -> str:
    """Extracts text from uploaded file based on its extension."""
    text = ""
    try:
        if file.filename.endswith(".pdf"):
            pdf_stream = io.BytesIO(file_contents)
            reader = PdfReader(pdf_stream)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file.filename.endswith(".docx"):
            docx_stream = io.BytesIO(file_contents)
            doc = Document(docx_stream)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.filename.endswith(".txt"):
            text = file_contents.decode("utf-8")
        elif file.filename.endswith((".xlsx", ".xls")):
            excel_stream = io.BytesIO(file_contents)
            xls = pd.ExcelFile(excel_stream)
            all_sheets_text = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                sheet_text = '\n'.join([' '.join(map(str, row)) for row in df.values])
                all_sheets_text.append(f"Sheet: {sheet_name}\n{sheet_text}")
            text = "\n\n---\n\n".join(all_sheets_text)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .txt, .pdf, .docx, or .xlsx file.")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"Could not extract any text from '{file.filename}'. The file might be empty or scanned as an image.")
            
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing file '{file.filename}': {e}")


# --- API Endpoints ---
@app.post("/session/clear", summary="Clear a session's chat history and knowledge base")
async def clear_session(request: SessionRequest):
    if request.session_id in sessions:
        sessions.pop(request.session_id, None)
        print(f"Cleared session: {request.session_id}")
    return {"status": "success", "message": "Session cleared."}


@app.post("/upload", summary="Upload a knowledge base file for a specific session")
async def upload_knowledge_base(session_id: str = Form(...), file: UploadFile = File(...)):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is missing.")
    if not embeddings:
        raise HTTPException(status_code=503, detail="Embeddings model is not available.")

    if session_id not in sessions:
        sessions[session_id] = {"vector_store": None, "rag_chain": None, "uploaded_files": []}
    
    session = sessions[session_id]

    if file.filename in session["uploaded_files"]:
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' has already been uploaded in this session.")

    try:
        file_contents = await file.read()
        text = extract_text_from_file(file, file_contents)
        
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not create any text chunks from the document.")
        
        if session["vector_store"]:
            new_docs_vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
            session["vector_store"].merge_from(new_docs_vector_store)
        else:
            session["vector_store"] = FAISS.from_texts(texts=chunks, embedding=embeddings)

        llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

        prompt_template = """
        You are a helpful AI assistant. Your goal is to be precise and concise.
        Use the following pieces of context from the user's documents to answer the question.
        Structure your answer clearly using Markdown with ## Headings and bullet points (*).
        After answering based on the context, you may provide helpful recommendations or suggest next steps if relevant.
        If you don't know the answer from the context, state clearly that the answer is not in the provided documents.
        
        Context: {context}
        
        Question: {question}
        Helpful Answer (formatted in Markdown):"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        session["rag_chain"] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=session["vector_store"].as_retriever(),
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        session["uploaded_files"].append(file.filename)
        file_list = ", ".join(session['uploaded_files'])
        return {"status": "success", "message": f"Added '{file.filename}'. Knowledge base now contains: {file_list}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/chat", response_model=ChatResponse, summary="Handle a chat message for a specific session")
async def chat_with_bot(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is missing.")

    # --- Intercept meta-questions BEFORE sending to the LLM ---
    
    # 1. Check for questions about uploaded documents
    doc_question_pattern = r"(what|which|list)\s+(file|doc|document|documet)s?\s+(have|did)\s+I\s+(upload|sent|send)|(list|show)\s+(me\s+)?my\s+(file|doc|document|documet)s?"
    if re.search(doc_question_pattern, request.question, re.IGNORECASE):
        if request.session_id in sessions and sessions[request.session_id].get("uploaded_files"):
            file_list = "\n".join([f"* {f}" for f in sessions[request.session_id]["uploaded_files"]])
            return ChatResponse(answer=f"## Uploaded Documents\nYou have uploaded the following documents in this session:\n{file_list}")
        else:
            return ChatResponse(answer="You have not uploaded any documents in this session yet.")

    # 2. Check for questions about the conversation history
    history_question_pattern = r"(what was|what's|what is)\s+(my|the)\s+(last|previous)\s+question|my\s+previous\s+question"
    if re.search(history_question_pattern, request.question, re.IGNORECASE):
        if request.chat_history:
            last_user_question = request.chat_history[-1][0]
            return ChatResponse(answer=f"Your previous question was: \"{last_user_question}\"")
        else:
            return ChatResponse(answer="You haven't asked any questions before this one.")

    # --- If not a meta-question, proceed to the LLM ---
    try:
        # Handle general conversation if no documents are uploaded
        if request.session_id not in sessions or not sessions[request.session_id].get("rag_chain"):
            if re.search(r"^\s*(hi|hello|hey)\s*$", request.question, re.IGNORECASE):
                return ChatResponse(answer="Hello! I'm ready to chat. For questions about specific documents, please upload a file first.")
            
            llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
            response = llm.invoke([
                SystemMessage(content="You are a helpful and friendly AI assistant."),
                HumanMessage(content=request.question)
            ])
            return ChatResponse(answer=response.content)

        # Use the RAG chain if documents have been uploaded
        session = sessions[request.session_id]
        result = session["rag_chain"].invoke({
            "question": request.question,
            "chat_history": request.chat_history
        })
        return ChatResponse(answer=result["answer"])
    except Exception as e:
        print(f"Error during chat for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
