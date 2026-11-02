"""
Dr. Asha — a personalized medical RAG assistant.

Architecture
------------
Every user turn runs through two LLM stages:

1. INTAKE (one fast JSON call)
   Extracts durable personal facts into a session profile, decides whether we
   can give a materially *personalized* answer now, or whether 1-3 clarifying
   questions would change the answer. Clarifications are returned to the UI as
   quick-reply chips / short text prompts, never as generic filler.

2. ANSWER (streamed)
   Retrieval over the session's uploaded reports (FAISS) + the user profile +
   everything the user clarified, answered in the Dr. Asha persona as an SSE
   token stream.

Sessions live in memory only (privacy by default) and are evicted after a TTL.
"""

import io
import json
import logging
import os
import re
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from docx import Document as DocxDocument
import pandas as pd
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "fastembed")  # fastembed | cohere
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-light-v3.0")

MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
MAX_FILES_PER_SESSION = int(os.getenv("MAX_FILES_PER_SESSION", "8"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "180"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))
MAX_CLARIFY_ROUNDS = int(os.getenv("MAX_CLARIFY_ROUNDS", "2"))
CHAT_HISTORY_WINDOW = int(os.getenv("CHAT_HISTORY_WINDOW", "12"))

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls")

ALLOWED_ORIGINS = [
    "https://asha-med.vercel.app",
    "https://document-chatbot-kambaa.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("asha")

# --------------------------------------------------------------------------
# App state (filled during startup)
# --------------------------------------------------------------------------

embeddings = None
sessions: Dict[str, Dict[str, Any]] = {}

PROFILE_EMPTY: Dict[str, Any] = {
    "name": None,
    "age": None,
    "sex": None,
    "pregnant": None,
    "diet": None,
    "conditions": [],
    "medications": [],
    "allergies": [],
    "symptoms": [],
    "symptom_duration": None,
    "preferences": [],
    "notes": [],
}


def _now() -> float:
    return time.time()


def _new_session() -> Dict[str, Any]:
    return {
        "created_at": _now(),
        "last_active": _now(),
        "vector_store": None,
        "files": [],            # [{name, chars, chunks, hash}]
        "file_gists": {},       # name -> 1-2 line gist shown to the intake step
        "profile": json.loads(json.dumps(PROFILE_EMPTY)),
        "pending": None,        # {"question", "questions", "round", "asked_at"}
        "history": [],          # [(user, bot)]
        "lock": asyncio.Lock(),
    }


def _merge_profile(profile: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge intake-extracted facts into the profile. Lists extend with unique
    values; scalars overwrite; explicit nulls are ignored."""
    list_keys = {"conditions", "medications", "allergies", "symptoms", "preferences", "notes"}
    for key, value in (updates or {}).items():
        if key not in PROFILE_EMPTY or value in (None, "", []):
            continue
        if key in list_keys:
            current = profile.setdefault(key, [])
            items = value if isinstance(value, list) else [value]
            for item in items:
                clean = str(item).strip()
                if clean and clean.lower() not in [c.lower() for c in current]:
                    current.append(clean)
        else:
            profile[key] = value
    return profile


# --------------------------------------------------------------------------
# Lifespan: embeddings + session janitor
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings
    log.info("Startup: initializing embeddings (%s)...", EMBEDDINGS_PROVIDER)
    if EMBEDDINGS_PROVIDER == "cohere":
        from langchain_cohere import CohereEmbeddings
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            raise RuntimeError("COHERE_API_KEY not set for EMBEDDINGS_PROVIDER=cohere")
        embeddings = CohereEmbeddings(cohere_api_key=cohere_key, model=COHERE_EMBED_MODEL)
        _ = embeddings.embed_query("warmup")
    else:
        from langchain_community.embeddings import FastEmbedEmbeddings
        embeddings = FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)
        _ = embeddings.embed_query("warmup")
    log.info("Embeddings ready: %s", EMBEDDINGS_PROVIDER)

    task = asyncio.create_task(_session_janitor())
    log.info("Application ready.")
    try:
        yield
    finally:
        task.cancel()
        log.info("Shutdown complete.")


async def _session_janitor():
    """Evict idle sessions and cap the total number of live sessions."""
    while True:
        try:
            await asyncio.sleep(600)
            cutoff = _now() - SESSION_TTL_MINUTES * 60
            expired = [sid for sid, s in sessions.items() if s["last_active"] < cutoff]
            for sid in expired:
                sessions.pop(sid, None)
            if len(sessions) > MAX_SESSIONS:
                by_age = sorted(sessions.items(), key=lambda kv: kv[1]["last_active"])
                for sid, _ in by_age[: len(sessions) - MAX_SESSIONS]:
                    sessions.pop(sid, None)
            if expired or len(sessions) >= MAX_SESSIONS:
                log.info("Janitor: evicted %d sessions, %d live", len(expired), len(sessions))
        except asyncio.CancelledError:
            break
        except Exception:
            log.exception("Session janitor error")


app = FastAPI(
    title="Dr. Asha Backend",
    description="Personalized medical RAG chatbot with an intake/clarification engine.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=4000)
    chat_history: List[Any] = Field(default_factory=list, description="Optional; server keeps its own history")


class ProfileIn(BaseModel):
    session_id: str
    profile: Dict[str, Any]


# --------------------------------------------------------------------------
# Groq helpers
# --------------------------------------------------------------------------

def make_llm(temperature: float, json_mode: bool = False, max_tokens: Optional[int] = None) -> ChatGroq:
    extra: Dict[str, Any] = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
        timeout=90,
        max_retries=2,
        max_tokens=max_tokens,
        reasoning_effort="low" if GROQ_MODEL.startswith("openai/gpt-oss") else None,
        model_kwargs=extra,
    )


def parse_json_answer(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a model response."""
    if not raw:
        return None
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("Intake returned unparseable JSON: %.200s", raw)
        return None


# --------------------------------------------------------------------------
# Document parsing
# --------------------------------------------------------------------------

def extract_text(filename: str, contents: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(contents))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if lower.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(contents))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(parts)
    if lower.endswith(".txt"):
        return contents.decode("utf-8", errors="replace")
    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents), header=None, dtype=str, on_bad_lines="skip")
        return "\n".join(" | ".join(v for v in row if v) for row in df.values.tolist())
    if lower.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(contents))
        sheets = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
            rows = [" | ".join("" if pd.isna(v) else str(v) for v in row) for row in df.values]
            sheets.append(f"Sheet: {sheet}\n" + "\n".join(rows))
        return "\n\n---\n\n".join(sheets)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{filename}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
    )


# --------------------------------------------------------------------------
# Stage 1 — Intake (profile extraction + clarify/answer decision)
# --------------------------------------------------------------------------

INTAKE_SYSTEM = """You are the INTAKE module of "Dr. Asha", a warm, careful virtual health assistant.
Today is {today}.

You receive: the user's PROFILE (may be mostly empty), a REPORT_MAP describing uploaded medical
reports, PENDING questions we asked in the previous turn, recent chat history, and the user's NEW
MESSAGE. You do three jobs in one JSON response:

JOB 1 — profile_updates: extract durable personal facts the user stated or implied in their NEW
MESSAGE (or as answers to pending questions). Allowed keys: name, age, sex ("male"/"female"),
pregnant (true/false), diet ("vegetarian"|"non-vegetarian"|"vegan"|"eggetarian"), conditions[],
medications[], allergies[], symptoms[], symptom_duration, preferences[], notes[]. Only facts
actually stated — never guess. Empty object {{}} if nothing new.

JOB 2 — action: decide "answer" or "clarify".
- "answer" when we can already give a materially personalized, safe answer (the report, the profile
  and the message contain what matters).
- "clarify" ONLY when 1-3 missing details would MATERIALLY change a safe, personalized answer —
  e.g. age/sex needed to interpret lab ranges, pregnancy, relevant medications, symptom duration or
  severity, or what the user wants out of the answer.
- FOOD, DIET & LIFESTYLE QUESTIONS ARE PERSONAL: for anything about meals, what to eat, weight,
  exercise or daily routine, these details change the whole answer and MUST be asked when not
  already in the profile: diet preference (vegetarian / non-vegetarian / vegan / eggetarian — do
  not assume one), allergies or foods to avoid (only ask if PROFILE.allergies is empty), and one
  of: cuisine preference, cooking time available, or a specific goal. Never recommend foods that
  clash with a known allergy or diet preference.
- Never ask for information already present in the PROFILE, the REPORT_MAP, or the message itself.
  Never ask clarifying questions for simple lookups ("what does X mean", "what was my hemoglobin
  value", "summarize my report").
- A pending question round counts: total clarification rounds per topic must not exceed {max_rounds}.
  If pending.round >= {max_rounds} OR the user's message is plausibly a reply to the pending
  questions, you MUST choose "answer" and fold whatever was answered into profile_updates.

JOB 3 — questions (only when action = "clarify"): 1 to 3 warm, specific, patient-friendly questions.
Each: {{"id": short_slug, "question": one warm sentence, "type": "choice"|"text",
"options": [2-4 short options] (choice only), "optional": true|false}}.
Prefer "choice" with sensible clinical options. Each question must state why it helps, briefly.

Also return "preamble": one short empathetic sentence shown above the questions (or "" when answering).

Respond with ONLY a JSON object, exactly this shape:
{{
  "profile_updates": {{...}},
  "action": "answer" | "clarify",
  "preamble": "...",
  "questions": [...]
}}"""


def build_intake_messages(
    profile: Dict[str, Any],
    file_gists: Dict[str, str],
    pending: Optional[Dict[str, Any]],
    history: List,
    question: str,
    has_reports: bool,
) -> List:
    recent = "\n".join(f"User: {q}\nDr. Asha: {a}" for q, a in history[-4:]) or "(first message)"
    report_map = (
        "\n".join(f"- {name}: {gist}" for name, gist in file_gists.items())
        if file_gists
        else "(no reports uploaded yet)"
    )
    sys = INTAKE_SYSTEM.format(
        today=datetime.now(timezone.utc).strftime("%d %b %Y"),
        max_rounds=MAX_CLARIFY_ROUNDS,
    )
    user = f"""PROFILE: {json.dumps(profile)}
REPORT_MAP: {report_map if has_reports else '(no reports uploaded)'}
PENDING_QUESTIONS: {json.dumps(pending) if pending else 'null'}
RECENT_CHAT:
{recent}

USER'S NEW MESSAGE: {question}

Return the JSON now."""
    return [SystemMessage(content=sys), HumanMessage(content=user)]


async def run_intake(session: Dict[str, Any], question: str) -> Dict[str, Any]:
    messages = build_intake_messages(
        session["profile"], session["file_gists"], session["pending"],
        session["history"], question, bool(session["vector_store"]),
    )
    llm = make_llm(temperature=0.1, json_mode=True, max_tokens=1200)
    try:
        response = await asyncio.to_thread(llm.invoke, messages)
        data = parse_json_answer(response.content)
    except Exception:
        log.exception("Intake call failed")
        data = None
    if not data or "action" not in data:
        # Safe fallback: answer with what we have rather than blocking the user.
        data = {"profile_updates": {}, "action": "answer", "preamble": "", "questions": []}
    return data


# --------------------------------------------------------------------------
# Stage 2 — Answer (streamed, RAG when reports exist)
# --------------------------------------------------------------------------

ANSWER_SYSTEM = """You are "Dr. Asha", a warm, empathetic, highly knowledgeable virtual health assistant.
Today is {today}.

You are speaking with ONE specific person. Use everything below to make your answer about THEM, not
about an average patient.

ABOUT THIS PERSON (profile, may be incomplete — never invent values):
{profile}

WHAT THEY CLARIFIED THIS TURN (fold this in; do not re-ask):
{clarifications}

UPLOADED REPORTS (retrieved excerpts; cite values exactly as written, with units and reference
ranges when shown):
{context}

RULES:
1. Personalize. If you know their age, sex, diet preference, conditions, medications or symptoms,
   use them explicitly ("Because you're 45, on metformin, and vegetarian, ..."). If the profile is
   thin, say what a typical range means and invite them to share the missing detail next.
2. Ground in the reports. Quote the exact result values and reference ranges from the excerpts.
   If the answer is not in the excerpts, say so plainly and offer general, clearly-labeled general
   information. NEVER invent a value.
3. Honor hard constraints: never suggest foods that clash with their diet preference or allergies,
   and tailor every food, exercise or lifestyle suggestion to the profile.
4. Structure with markdown: a 1-2 sentence warm opening that directly addresses them; then short
   sections such as **What your report says**, **What this means for you**, **What you can do next**,
   and **Good questions for your doctor** (2-3 concrete questions).
5. Explain like to a patient: define any medical term the first time in plain words.
6. Never give a definitive diagnosis. Say what results are associated with, and recommend their
   doctor for diagnosis and treatment. If any symptom suggests an emergency (crushing chest pain,
   trouble breathing, sudden weakness, heavy bleeding...), tell them to seek urgent care immediately.
7. Do not re-ask questions they just answered.
7. End with exactly this line on its own paragraph:
   I'm an AI assistant, not a substitute for professional medical advice — please confirm anything
   important with your doctor or a qualified healthcare professional."""

GENERAL_SYSTEM = """You are "Dr. Asha", a warm, empathetic virtual health assistant. The user has NOT
uploaded a report yet, so you cannot see their results.

ABOUT THIS PERSON (profile, may be incomplete — never invent values):
{profile}

WHAT THEY CLARIFIED THIS TURN:
{clarifications}

RULES:
1. Answer in a personalized way using whatever the profile and message contain ("Because you're
   32 and pregnant, ...").
2. You may share general, clearly-labeled health education — but always point out that their own
   doctor and their own reports are the reliable source for specifics, and invite them to upload a
   report (PDF, DOCX, TXT, CSV or Excel) for a grounded review.
3. Keep it focused: a warm opening, a short clear explanation, one or two practical next steps, and
   2-3 "Good questions for your doctor" when relevant.
4. Never give a definitive diagnosis. Describe possibilities and associations, never conclusions.
   If a symptom suggests an emergency, tell them to seek urgent care immediately.
5. End with exactly this line on its own paragraph:
   I'm an AI assistant, not a substitute for professional medical advice — please confirm anything
   important with your doctor or a qualified healthcare professional."""


def build_answer_messages(session: Dict[str, Any], question: str, clarify_notes: str) -> List:
    profile_json = json.dumps(session["profile"], indent=2)
    if session["vector_store"]:
        docs = session["vector_store"].as_retriever(
            search_type="mmr", search_kwargs={"k": 6, "fetch_k": 24}
        ).invoke(question)
        context = "\n\n---\n\n".join(
            f"[from {d.metadata.get('source', 'report')}]\n{d.page_content}" for d in docs
        ) or "(no relevant excerpts found)"
        system = ANSWER_SYSTEM.format(
            today=datetime.now(timezone.utc).strftime("%d %b %Y"),
            profile=profile_json,
            clarifications=clarify_notes or "(none)",
            context=context,
        )
    else:
        system = GENERAL_SYSTEM.format(
            today=datetime.now(timezone.utc).strftime("%d %b %Y"),
            profile=profile_json,
            clarifications=clarify_notes or "(none)",
        )
    # Include recent turns so follow-ups ("what about the second one?") make sense.
    messages = [SystemMessage(content=system)]
    for q, a in session["history"][-4:]:
        messages.append(HumanMessage(content=q))
        messages.append(SystemMessage(content=f"(previous reply) {a}"))
    messages.append(HumanMessage(content=question))
    return messages


def _clarify_notes_from(pending: Optional[Dict[str, Any]], message: str) -> str:
    if not pending:
        return ""
    asked = "; ".join(q.get("question", "") for q in pending.get("questions", []))
    return f"The user was previously asked: {asked}. Their reply: {message}"


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "embeddings": EMBEDDINGS_PROVIDER,
        "live_sessions": len(sessions),
    }


@app.post("/api/upload")
async def upload_report(session_id: str = Form(...), file: UploadFile = File(...)):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is missing.")
    if not embeddings:
        raise HTTPException(status_code=503, detail="Embeddings not ready yet — retry shortly.")

    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is not a supported type. Use: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_FILE_MB} MB limit.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    session = sessions.setdefault(session_id, _new_session())
    async with session["lock"]:
        session["last_active"] = _now()
        content_hash = uuid.uuid5(uuid.NAMESPACE_URL, filename + str(len(contents))).hex

        if any(f["hash"] == content_hash for f in session["files"]):
            raise HTTPException(status_code=409, detail=f"'{filename}' is already part of this session.")
        if len(session["files"]) >= MAX_FILES_PER_SESSION:
            raise HTTPException(
                status_code=400,
                detail=f"Session limit of {MAX_FILES_PER_SESSION} files reached. Start a new chat to reset.",
            )

        try:
            text = extract_text(filename, contents)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Failed to parse %s", filename)
            raise HTTPException(status_code=422, detail=f"Could not read '{filename}': {exc}") from exc
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail=f"No text could be extracted from '{filename}'. Scanned/image-only PDFs are not supported.",
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = splitter.split_text(text)
        if not chunks:
            raise HTTPException(status_code=422, detail=f"'{filename}' produced no readable chunks.")

        store = FAISS.from_texts(chunks, embedding=embeddings, metadatas=[{"source": filename}] * len(chunks))
        if session["vector_store"]:
            session["vector_store"].merge_from(store)
        else:
            session["vector_store"] = store

        # A gist of this report feeds the intake step's topic awareness.
        head = text[:3000]
        try:
            gist_llm = make_llm(temperature=0.1, max_tokens=120)
            gist = await asyncio.to_thread(
                gist_llm.invoke,
                [HumanMessage(content=(
                    "In ONE sentence, state what kind of medical document this is and list the most "
                    "notable flagged or abnormal values (with values). Text:\n" + head
                ))],
            )
            session["file_gists"][filename] = gist.content.strip()
        except Exception:
            log.exception("Gist generation failed for %s", filename)
            session["file_gists"][filename] = f"{filename} ({len(chunks)} sections)"

        # A patient-friendly summary shown immediately after upload.
        summary = ""
        try:
            summary_llm = make_llm(temperature=0.4, max_tokens=400)
            summary_resp = await asyncio.to_thread(
                summary_llm.invoke,
                [HumanMessage(content=(
                    "A patient just uploaded this medical report. In 3-5 short sentences, warmly explain "
                    "what kind of report it is and what stands out most (flag high/low values with their "
                    "numbers). Plain language, no diagnosis, end by inviting a question. Text:\n" + head
                ))],
            )
            summary = summary_resp.content.strip()
        except Exception:
            log.exception("Summary generation failed for %s", filename)

        session["files"].append({"name": filename, "hash": content_hash,
                                 "chars": len(text), "chunks": len(chunks)})
        names = ", ".join(f["name"] for f in session["files"])

    return {
        "status": "success",
        "message": f"Added **{filename}** — I can now see: {names}.",
        "summary": summary,
        "files": [f["name"] for f in session["files"]],
        "profile": session["profile"],
    }


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    session = sessions.get(request.session_id)
    if session is None:
        session = sessions.setdefault(request.session_id, _new_session())
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    async def event_stream():
        started = _now()
        session["last_active"] = _now()
        try:
            yield _sse({"type": "status", "stage": "intake",
                        "message": "Dr. Asha is reviewing your question…"})

            intake = await run_intake(session, question)
            async with session["lock"]:
                _merge_profile(session["profile"], intake.get("profile_updates"))

            if intake.get("action") == "clarify" and intake.get("questions") and \
                    (session["pending"] is None or session["pending"].get("round", 0) < MAX_CLARIFY_ROUNDS):
                async with session["lock"]:
                    session["pending"] = {
                        "question": question,
                        "questions": intake["questions"],
                        "round": (session["pending"] or {}).get("round", 0) + 1,
                        "asked_at": _now(),
                    }
                yield _sse({"type": "clarify",
                            "preamble": intake.get("preamble", ""),
                            "questions": intake["questions"],
                            "profile": session["profile"]})
                yield _sse({"type": "done", "mode": "clarify",
                            "took_ms": int((_now() - started) * 1000),
                            "profile": session["profile"]})
                return

            pending = session["pending"]
            clarify_notes = _clarify_notes_from(pending, question)
            async with session["lock"]:
                session["pending"] = None

            if session["vector_store"]:
                yield _sse({"type": "status", "stage": "retrieve",
                            "message": "Reading your reports…"})

            llm = make_llm(temperature=0.55)
            messages = build_answer_messages(session, question, clarify_notes)
            yield _sse({"type": "answer_start"})

            answer_parts: List[str] = []
            for chunk in llm.stream(messages):
                piece = chunk.content if isinstance(chunk.content, str) else "".join(
                    p.get("text", "") for p in chunk.content if isinstance(p, dict))
                if piece:
                    answer_parts.append(piece)
                    yield _sse({"type": "token", "value": piece})

            answer = "".join(answer_parts)
            async with session["lock"]:
                session["history"].append((question, answer))
                if len(session["history"]) > CHAT_HISTORY_WINDOW:
                    session["history"] = session["history"][-CHAT_HISTORY_WINDOW:]
            yield _sse({"type": "done", "mode": "rag" if session["vector_store"] else "general",
                        "took_ms": int((_now() - started) * 1000),
                        "profile": session["profile"]})
        except HTTPException as exc:
            yield _sse({"type": "error", "message": exc.detail})
        except Exception as exc:
            log.exception("Chat stream failed")
            yield _sse({"type": "error", "message": f"Something went wrong: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Non-streaming variant: runs the same pipeline and returns the final message."""
    session = sessions.get(request.session_id)
    if session is None:
        session = sessions.setdefault(request.session_id, _new_session())
    question = request.question.strip()
    intake = await run_intake(session, question)
    async with session["lock"]:
        _merge_profile(session["profile"], intake.get("profile_updates"))
        if intake.get("action") == "clarify" and intake.get("questions") and \
                (session["pending"] is None or session["pending"].get("round", 0) < MAX_CLARIFY_ROUNDS):
            session["pending"] = {
                "question": question,
                "questions": intake["questions"],
                "round": (session["pending"] or {}).get("round", 0) + 1,
                "asked_at": _now(),
            }
            return {"type": "clarify", "preamble": intake.get("preamble", ""),
                    "questions": intake["questions"], "profile": session["profile"]}
        pending, session["pending"] = session["pending"], None

    clarify_notes = _clarify_notes_from(pending, question)
    llm = make_llm(temperature=0.55)
    response = await asyncio.to_thread(llm.invoke, build_answer_messages(session, question, clarify_notes))
    answer = response.content
    async with session["lock"]:
        session["history"].append((question, answer))
    return {"type": "answer", "answer": answer, "profile": session["profile"]}


@app.get("/api/profile")
async def get_profile(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"profile": session["profile"], "files": [f["name"] for f in session["files"]]}


@app.post("/api/profile")
async def update_profile(body: ProfileIn):
    session = sessions.setdefault(body.session_id, _new_session())
    async with session["lock"]:
        _merge_profile(session["profile"], body.profile)
        session["last_active"] = _now()
        return {"profile": session["profile"]}


@app.post("/api/session/clear")
async def clear_session(body: Dict[str, str]):
    sid = body.get("session_id", "")
    removed = sessions.pop(sid, None)
    log.info("Session cleared: %s (found=%s)", sid[:8], bool(removed))
    return {"status": "success"}


# Serve the frontend from this origin too (handy for local dev and single-service deploys).
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
