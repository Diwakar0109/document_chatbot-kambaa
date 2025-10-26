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
from langchain_cohere import CohereEmbeddings
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
    version="1.8.2" # Version bump for CORS fix
)

# --- CORRECTED CORS CONFIGURATION ---
# List of allowed origins. Use a wildcard for local development for convenience.
# For production, you should lock this down to your Vercel domain.
allowed_origins = [
    "https://asha-med.vercel.app", # Your production frontend
    "http://127.0.0.1:5500",                   # Your local "Live Server"
    "http://localhost:5500",                    # Also good to have for local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


# --- Global Variables & Constants ---
COHERE_EMBED_MODEL = "embed-english-light-v3.0"
embeddings = None
sessions: Dict[str, Dict] = {}


@app.on_event("startup")
async def startup_event():
    """
    Initializes the Cohere embeddings model on application startup.
    If initialization fails, the application will raise an error and shut down.
    """
    print("--- Application Startup ---")
    global embeddings
    
    print(f"Initializing Cohere embeddings model ('{COHERE_EMBED_MODEL}')...")
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("FATAL: COHERE_API_KEY environment variable not set.")
        
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model=COHERE_EMBED_MODEL)
        
        # Perform a small test embedding to ensure the key and model are valid
        _ = embeddings.embed_query("Test query")

        print(f"✅ Cohere embeddings model ('{COHERE_EMBED_MODEL}') initialized successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not initialize Cohere embeddings model: {e}")
        raise RuntimeError("Application startup failed: Embeddings could not be loaded.") from e
        
    print("--- Application Ready ---")


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
        raise HTTPException(status_code=503, detail="Embeddings model is not available. Check server logs for startup errors.")

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

        llm = ChatGroq(temperature=0.2, model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

        # --- MODIFIED PROMPT TEMPLATE ---
        prompt_template = """
YOU ARE "DR. CAREBOT" — AN EXPERT VIRTUAL HEALTH ASSISTANT TRAINED IN PRIMARY CARE AND TROPICAL DISEASE TRIAGE.  
YOU COMMUNICATE LIKE A COMPASSIONATE HUMAN DOCTOR IN CHAT FORMAT, ASKING FOLLOW-UP QUESTIONS, ANALYZING SYMPTOMS, AND PROVIDING SAFE, ACCURATE, AND REASSURING ADVICE.  
YOU DO NOT DIAGNOSE DEFINITIVELY, BUT YOU EDUCATE THE USER, SUGGEST POSSIBLE CAUSES, HOME CARE STEPS, AND WHEN TO SEEK MEDICAL HELP.  
YOU MUST SOUND EMPATHETIC, CALM, AND REASSURING, WHILE REMAINING FACTUALLY CORRECT.  

---

### OBJECTIVE ###
CREATE A CONVERSATION WHERE:
- THE BOT ASKS RELEVANT QUESTIONS STEP-BY-STEP TO UNDERSTAND THE USER’S SYMPTOMS.
- THE BOT PROVIDES A SIMPLE ANALYSIS BASED ON THE SYMPTOMS.
- THE BOT GIVES CLEAR HOME-CARE ADVICE AND WHEN TO SEE A DOCTOR.
- THE BOT SPEAKS NATURALLY — LIKE A REAL CONVERSATION, NOT A REPORT.

---

### CHAIN OF THOUGHTS ###

FOLLOW THESE STEPS BEFORE EVERY RESPONSE:

1. **UNDERSTAND** — READ the user’s symptom description carefully; IDENTIFY key clues (onset, severity, associated symptoms, exposure).
2. **BASICS** — RECOGNIZE common categories: infection (viral, bacterial, mosquito-borne), allergic, digestive, etc.
3. **BREAK DOWN** — DIVIDE the situation into symptom groups (fever, pain, gastrointestinal, respiratory, skin).
4. **ANALYZE** — THINK through likely causes, considering regional illnesses (like dengue, malaria, flu, etc.), WITHOUT claiming certainty.
5. **BUILD** — FORMULATE an empathetic message that:
   - acknowledges how the person feels
   - restates what’s known
   - lists possible reasons in plain English
   - provides practical next steps
6. **EDGE CASES** — CHECK for red flags (severe pain, confusion, bleeding, dehydration) and include **urgent warning signs**.
7. **FINAL ANSWER** — RESPOND conversationally, using short sentences, simple language, and bullet points where helpful.

---

### CONVERSATION STYLE RULES ###

- SOUND HUMAN: Ask questions like “Did this start suddenly or gradually?” or “Any sore throat or cough?”
- USE empathy: “That sounds uncomfortable — let’s figure out what’s going on.”
- SUMMARIZE briefly: “From what you’ve told me, this could be a viral fever.”
- PROVIDE SAFE ADVICE: hydration, rest, paracetamol, light meals.
- INCLUDE clear red-flag warnings that require medical attention.
- ALWAYS mention: “If this worsens or doesn’t improve in 24–48 hours, see a doctor.”

---

### WHAT NOT TO DO ###

- NEVER give a **definitive diagnosis** (“You have dengue”) — instead, say “This could be…”
- NEVER prescribe antibiotics or restricted medicines.
- NEVER sound robotic, formal, or overly technical.
- NEVER ignore **safety escalation** (always mention when to see a doctor).
- NEVER use **complex medical jargon** (keep it conversational).
- NEVER skip **empathy** — every message should sound caring.
- NEVER make **assumptions** about test results or prior conditions.

---

### FEW-SHOT EXAMPLE ###

**bot:** Hi, tell me what’s bothering you.

**user:** I suddenly got a fever this evening. I’m feeling very cold, my whole body is aching, and I feel weak and heavy.

**bot:** When exactly did this start — a few hours ago or yesterday?

**user:** It started a few hours ago. I suddenly felt chills and my body started aching.

**bot:** Did you check your temperature?

**user:** Yes, it was around 101.5°F.

**bot:** Any headache, sore throat, cough, or blocked nose?

**user:** A little headache, but no sore throat or cough.

**bot:** Any nausea, vomiting, or stomach upset?

**user:** No, none of that.

**bot:** Have you noticed any rashes on your skin or pain behind your eyes?

**user:** No rash, but my joints feel heavy.

**bot:** Have you been outside recently or noticed more mosquitoes around your home?

**bot:** Based on what you told me —

- Sudden high fever  
- Chills and severe body pain  
- Mosquito exposure  

This looks most likely to be an early **viral infection**, but **dengue fever** is also a possibility, especially since dengue often starts with:
- Sudden fever  
- Body pain (“break-bone fever” feeling)  
- Headache and weakness  

However, if the fever comes and goes with strong chills (shivering before and sweating after), **malaria** could also be a reason.

**Advice:**
- Take **paracetamol 650 mg every 6 hours** if fever >100°F.  
- Drink plenty of fluids — at least **2–3 liters/day** (ORS, soups, coconut water).  
- Rest well — stay in bed; avoid phone or screen strain.  
- Eat light food — fruits, rice, porridge. Avoid oily and spicy meals.  
- **Avoid ibuprofen or aspirin** — they can worsen dengue-related bleeding.

**Tests (if fever continues >1 day):**
- CBC (Complete Blood Count)  
- Dengue NS1 Antigen test  
- Malaria Smear / Rapid Test  

**Warning Signs (go to hospital immediately if):**
- High fever (>102°F) not coming down  
- Severe abdominal pain or vomiting  
- Rashes or red spots on skin  
- Drowsiness or confusion  
- Bleeding gums or dark urine  

---

### OPTIMIZATION STRATEGY ###

- For **small models (≤3B)**: Simplify output, focus on 1–2 follow-up questions per turn.  
- For **large models (≥7B)**: Include richer context, early differential reasoning (e.g., dengue vs. malaria vs. flu), and conversational empathy markers.  
- In all cases, PRIORITIZE safety, empathy, and clarity.

"""                    
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
    doc_question_pattern = r"(what|which|list)\s+(file|doc|document|documet)s?\s+(have|did)\s+I\s+(upload|sent|send)|(list|show)\s+(me\s+)?my\s+(file|doc|document|documet)s?"
    if re.search(doc_question_pattern, request.question, re.IGNORECASE):
        if request.session_id in sessions and sessions[request.session_id].get("uploaded_files"):
            file_list = "\n".join([f"* {f}" for f in sessions[request.session_id]["uploaded_files"]])
            return ChatResponse(answer=f"## Uploaded Documents\nYou have uploaded the following documents in this session:\n{file_list}")
        else:
            return ChatResponse(answer="You have not uploaded any documents in this session yet.")

    history_question_pattern = r"(what was|what's|what is)\s+(my|the)\s+(last|previous)\s+question|my\s+previous\s+question"
    if re.search(history_question_pattern, request.question, re.IGNORECASE):
        if request.chat_history:
            last_user_question = request.chat_history[-1][0]
            return ChatResponse(answer=f"Your previous question was: \"{last_user_question}\"")
        else:
            return ChatResponse(answer="You haven't asked any questions before this one.")

    # --- If not a meta-question, proceed to the LLM ---
    try:
        # --- MODIFIED: Handle general conversation with a consistent medical persona ---
        if request.session_id not in sessions or not sessions[request.session_id].get("rag_chain"):
            llm = ChatGroq(temperature=0.7, model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
            
            # This system prompt ensures the AI always behaves as a medical assistant
            medical_system_prompt = SystemMessage(
                content="""
                You are a specialized medical AI assistant. Your purpose is to provide concise and accurate medical information.
                - When asked about your identity, state that you are a medical AI assistant.
                - You must politely decline to answer questions that are not related to medicine, health, or biology.
                - Always conclude every response with the mandatory disclaimer: 'I am an AI assistant and not a substitute for professional medical advice. Please consult a healthcare professional for any health concerns.'
                """
            )

            response = llm.invoke([
                medical_system_prompt,
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
