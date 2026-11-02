"""Edit project-final-review.pptx in place-clone: same format, Dr. Asha content, simple English.

Output: project-final-review_updated.pptx (original untouched).
Also draws the 3 diagram pictures (methodology flowchart + 2 design diagrams)
as SVG -> PNG in the friend's dark-navy poster style and swaps the image parts.
"""

import copy
import io
import shutil
from pathlib import Path

import pymupdf
from pptx import Presentation

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "project-final-review.pptx"
OUT = ROOT / "project-final-review_updated.pptx"
ASSETS = ROOT / "docs" / "tpl_assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- SVG diagrams

FONT = "Helvetica, Arial, sans-serif"
NAVY, PANEL = "#12263F", "#17324F"
BLUE_BG, BLUE_TX = "#BBDEFB", "#0D2B4E"
GREEN, TEAL, AMBER, RED = "#2E7D46", "#16697A", "#E8A33D", "#C25450"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, fill, lines, stroke="none", dash="0", rx=14, tsz=26, tcol="#FFFFFF"):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="3" stroke-dasharray="{dash}"/>']
    n = len(lines)
    for i, (txt, bold) in enumerate(lines):
        size = tsz if n == 1 else (tsz if i == 0 else tsz - 5)
        weight = "bold" if bold or i == 0 else "normal"
        ty = y + h / 2 + (i - (n - 1) / 2) * (size + 8) + size * 0.35
        out.append(f'<text x="{x + w/2}" y="{ty:.0f}" font-family="{FONT}" font-size="{size}" '
                   f'font-weight="{weight}" fill="{tcol}" text-anchor="middle">{esc(txt)}</text>')
    return "\n".join(out)


def arrow(x1, y1, x2, y2, label=None, lcol="#B8CCD4"):
    head = 14
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    ax = x2 - head * math.cos(ang); ay = y2 - head * math.sin(ang)
    p1 = (ax - 8 * math.cos(ang - 0.5), ay - 8 * math.sin(ang - 0.5))
    p2 = (ax - 8 * math.cos(ang + 0.5), ay - 8 * math.sin(ang + 0.5))
    out = [f'<line x1="{x1}" y1="{y1}" x2="{ax:.0f}" y2="{ay:.0f}" stroke="#FFFFFF" stroke-width="5"/>',
           f'<polygon points="{x2},{y2} {p1[0]:.0f},{p1[1]:.0f} {p2[0]:.0f},{p2[1]:.0f}" fill="#FFFFFF"/>']
    if label:
        out.append(f'<text x="{(x1+x2)/2:.0f}" y="{(y1+y2)/2 - 14:.0f}" font-family="{FONT}" font-size="22" '
                   f'font-style="italic" fill="{lcol}" text-anchor="middle">{esc(label)}</text>')
    return "\n".join(out)


def svg_doc(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="{NAVY}"/>{body}</svg>')


def flowchart():  # slide 9 — 11.43 x 6.02 in
    b = [f'<text x="950" y="66" font-family="{FONT}" font-size="42" font-weight="bold" fill="#FFFFFF" text-anchor="middle">HOW DR. ASHA ANSWERS EVERY QUESTION</text>']
    b.append(box(700, 100, 500, 84, BLUE_BG, [("USER ASKS A QUESTION", True)], tcol=BLUE_TX, tsz=27))
    b.append(arrow(950, 184, 950, 244))
    b.append(box(560, 244, 780, 168, GREEN, [
        ("STAGE 1 — INTAKE (one fast call)", True),
        ("Reads your profile + question,", False),
        ("decides: answer now, or ask first?", False)], tsz=27))
    b.append(box(60, 288, 420, 150, AMBER, [
        ("ASKS 1-3 QUICK CHIPS", True),
        ("Age? Veg or non-veg?", False),
        ("Any allergies?", False)], tcol="#3A2A05", tsz=25))
    b.append(arrow(480, 340, 560, 340, "ask"))
    b.append(arrow(270, 438, 270, 610, "your answers"))
    b.append(arrow(270, 610, 560, 610, None))
    b.append(box(1450, 244, 390, 320, PANEL, [("SESSION MEMORY (RAM)", True)], stroke="#FFFFFF", dash="10 8", tsz=24))
    for i, t in enumerate(["Profile facts", "Reports + FAISS index", "Chat history"]):
        b.append(box(1478, 300 + i * 74, 334, 58, AMBER, [(t, False)], tcol="#3A2A05", tsz=20, rx=9))
    b.append(f'<text x="1645" y="545" font-family="{FONT}" font-size="19" fill="#B8CCD4" text-anchor="middle">cleared after 3 hours idle</text>')
    b.append(arrow(950, 412, 950, 500, "ready"))
    b.append(box(560, 500, 780, 160, TEAL, [
        ("STAGE 2 — PERSONALIZED ANSWER", True),
        ("Finds your report values (FAISS),", False),
        ("weaves in profile + your clarifications", False)], tsz=27))
    b.append(arrow(950, 660, 950, 720))
    b.append(box(560, 720, 780, 120, RED, [
        ("REPLY STREAMS LIVE + MEDICAL DISCLAIMER", True)], tsz=26))
    return svg_doc(1900, 1000, "\n".join(b))


def architecture():  # slide 13 — 5.22 x 4.87 in
    b = [f'<text x="535" y="60" font-family="{FONT}" font-size="36" font-weight="bold" fill="#FFFFFF" text-anchor="middle">SYSTEM ARCHITECTURE</text>']
    b.append(box(120, 92, 830, 150, BLUE_BG, [
        ("BROWSER — CHAT UI", True),
        ("JavaScript · SSE stream · quick-reply chips", False)], tcol=BLUE_TX, tsz=27))
    b.append(arrow(535, 242, 535, 320, "SSE"))
    b.append(box(120, 320, 830, 210, GREEN, [
        ("FASTAPI BACKEND", True),
        ("Intake + answer LLM calls · FAISS search", False),
        ("In-memory sessions · upload guards", False)], tsz=28))
    b.append(arrow(535, 530, 535, 620, "chat API"))
    b.append(box(120, 620, 830, 160, "#23405E", [
        ("GROQ LPU — gpt-oss-120b", True),
        ("The only place report text ever goes", False)], tsz=27))
    b.append(arrow(535, 780, 535, 860, "local"))
    b.append(box(120, 860, 830, 110, AMBER, [
        ("FASTEMBED (ONNX) — embeddings made on your machine", True)], tcol="#3A2A05", tsz=24))
    return svg_doc(1070, 1000, "\n".join(b))


def session_design():  # slide 14 — 6.39 x 5.18 in
    b = [f'<text x="617" y="60" font-family="{FONT}" font-size="36" font-weight="bold" fill="#FFFFFF" text-anchor="middle">SESSION DATA DESIGN</text>']
    b.append(f'<rect x="60" y="90" width="1115" height="600" rx="18" fill="{PANEL}" stroke="#5EEAD4" stroke-width="3" stroke-dasharray="12 8"/>')
    b.append(f'<text x="617" y="140" font-family="{FONT}" font-size="26" font-weight="bold" fill="#5EEAD4" text-anchor="middle">ONE SESSION — LIVES IN RAM ONLY</text>')
    b.append(box(100, 165, 1035, 140, AMBER, [
        ("PROFILE", True),
        ("age · sex · diet · medicines · allergies · symptoms", False)], tcol="#3A2A05", tsz=26))
    b.append(box(100, 325, 1035, 140, AMBER, [
        ("REPORTS", True),
        ("file list + FAISS vector store (per session)", False)], tcol="#3A2A05", tsz=26))
    b.append(box(100, 485, 1035, 140, AMBER, [
        ("PENDING", True),
        ("clarify questions asked · round count (max 2)", False)], tcol="#3A2A05", tsz=26))
    b.append(arrow(617, 750, 617, 655, "updates"))
    b.append(box(160, 755, 915, 130, GREEN, [
        ("INTAKE updates PROFILE every turn ·", True),
        ("ANSWER uses PROFILE + REPORTS", True)], tsz=26))
    b.append(f'<text x="617" y="955" font-family="{FONT}" font-size="24" fill="#B8CCD4" text-anchor="middle">A janitor clears sessions idle for 3 hours.</text>')
    return svg_doc(1235, 1000, "\n".join(b))


def raster(svg_text, w_in, h_in, fmt):
    p = ASSETS / "tmp.svg"
    p.write_text(svg_text, encoding="utf-8")
    doc = pymupdf.open(str(p))
    zoom = (w_in * 72) / doc[0].rect.width
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    return pix.tobytes("jpg" if fmt == "jpeg" else "png")


# ---------------------------------------------------------------- text helpers

def _norm(s):
    return s.replace("\x0b", "").replace("\r", "").strip()


def replace_in_paragraph(p, new_text):
    runs = p.runs
    if not runs:
        p.add_run().text = new_text
        return
    runs[0].text = new_text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def apply_map(tf, mapping, tag):
    m = {_norm(k): v for k, v in mapping.items()}
    used = set()
    for p in tf.paragraphs:
        key = _norm(p.text)
        if key in m:
            replace_in_paragraph(p, m[key])
            used.add(key)
    for key in m:
        if key not in used:
            print(f"  !! unmatched [{tag}]: {key[:70]!r}")


def iter_frames(shapes):
    for sh in shapes:
        if sh.shape_type == 6:
            yield from iter_frames(sh.shapes)
        elif sh.has_text_frame:
            yield sh


def swap_picture(prs, slide_idx, shape_id, svg, w_in, h_in):
    slide = prs.slides[slide_idx]
    for sh in slide.shapes:
        if sh.shape_id == shape_id and sh.shape_type == 13:
            rId = sh._element.blip_rId
            part = slide.part.related_part(rId)
            ext = str(part.partname).rsplit(".", 1)[-1].lower()
            data = raster(svg, w_in, h_in, ext)
            print(f"  swapped S{slide_idx+1} picture -> {part.partname} ({len(data)//1024} KB {ext})")
            part._blob = data
            return
    print(f"  !! picture id={shape_id} not found on slide {slide_idx+1}")


# ---------------------------------------------------------------- content maps

M = {}  # (slide_idx, shape_id) -> {old_norm: new}

M[(0, 89)] = {"COMMUNITY PLATFORM": "AI MEDICAL CHATBOT"}
M[(0, 91)] = {"MAGESH KANNAN K (232AD189)": "DIWAKAR R (roll no.)"}

S4 = {
    "To design and develop the Nexus Community Platform\u2014an enterprise-grade, decoupled technical community platform combining an AI-powered problem-solving engine, interactive micro-apps, and a resilient Dual-Mode runtime engine.":
        "To design and develop Dr. Asha\u2014an AI health companion that reads a patient\u2019s medical report and answers with personal context. It asks short questions first (age, diet, allergies), so every answer fits the person, not an average patient.",
    "Implement stateless JWT Authentication, Redis token blacklisting, and strict RBAC via RolesMaster UUIDs.":
        "Build a session profile that quietly stores age, sex, diet, medicines and allergies from normal chat.",
    "Build  AI Modules: AI Technical Copilot (/assistant), Semantic Vector Search, and Duplicate Detection and Expert recommendation system.":
        "Add an intake step that checks each question and asks 1-3 quick chip questions when details are missing.",
    "Automate Post Classification & Urgency Triage (Critical to Low) with Verified Solutions and an Unanswered Queue.":
        "Ground every answer in the uploaded report (PDF, DOCX, TXT, CSV, Excel) using FAISS retrieval.",
    "Deliver interactive micro-apps: nested Markdown discussions,  events with Google Meet RSVPs, live polls, and surveys.":
        "Stream answers in a warm doctor persona with exact values, reference ranges and a fixed medical disclaimer.",
}
M[(3, 117)] = S4

S5 = {
    "AI Problem-Solving & Triage Engine": "Personal Intake & Clarification",
    "AI Copilot (/assistant), Cosine-Similarity Semantic Vector Search, and >75% Duplicate Detection.":
        "One fast JSON call extracts profile facts and decides: answer now or ask first (max 2 rounds).",
    "Automated Post Classification, Production Urgency Scoring, and Verified Solutions.":
        "Quick-reply chips for diet, allergies and symptom details \u2014 almost no typing needed.",
    "Expert recommendation System for unanswered questions":
        "Answers reuse the profile, so nothing is asked twice.",
    "Community Lifecycle & RBAC Governance": "Report Intelligence & Streaming",
    "End-to-end workspace governance, founder/admin member controls (verifyAdminAccess), and Unanswered Triage Queue (/unanswered).":
        "Parse PDF, DOCX, TXT, CSV and Excel; chunk, embed locally and index per session in FAISS.",
    "Interactive Micro-Apps & Dual-Mode Runtime": "Privacy, Guards & Live UI",
    "Nested Markdown threads, AMA events with waitlist RSVPs, live polls, multi-question JSON surveys, and AWS S3 resource vault.":
        "Memory-only sessions evicted after 3 hours; local embeddings; sanitized markdown; live SSE streaming.",
}
M[(4, 126)] = S5

S6 = {
    "Fragmented Technical Collaboration: Traditional forums lack integrated live polling, multi-question surveys, Google Meet AMA RSVPs, and AWS S3 code repositories in one unified hub.":
        "Generic answers: most health bots reply with textbook text \u2014 the same words for a teenager and a 75-year-old patient, ignoring medicines, diet and history.",
    "Duplicate Questions & Unresolved Threads: Engineers repeatedly solve identical issues; our Novelty Duplicate Detection (>75% match), Urgency Engine, and Unanswered Triage Queue resolve this gap.":
        "One-shot guessing: bots rarely ask the two details that would change the advice \u2014 age, diet, allergies \u2014 so people get general text instead of personal guidance.",
    "Keyword Search Limitations: Standard text search misses conceptual context, necessitating our Semantic Vector Search (Cosine Similarity), AI Technical Copilot, and Verified Solutions.":
        "Reports stay unused: bots that cannot read lab values turn \u201Cwhat does my HbA1c mean?\u201D into a definition, not a personalized, report-grounded explanation.",
}
M[(5, 135)] = S6

S7 = {
    "Modern Type-Safe Stack: Next.js 16 (App Router), React 19, Express.js 5, and TypeScript 5.8.":
        "Proven Stack: FastAPI backend, LangChain pipeline and a plain JavaScript frontend.",
    "Prisma 7 ORM & Validation: Open-source Prisma 7 (25+ PostgreSQL tables), Zod 4.3 validators, Redis cache, and node-cron jobs.":
        "Local Embeddings: FastEmbed ONNX (bge-small) runs on the machine itself \u2014 no API key, no third party.",
    " AI & Dual-Mode Engine: Cosine-similarity vector search, urgency classifiers, and  Expert recomm":
        "Fast Model Hosting: Groq serves gpt-oss-120b at ~414 tokens per second with low reasoning effort.",
    "Zero Licensing Cost: Open-source Next.js, Express, Prisma, PostgreSQL, and Tailwind CSS v4.":
        "Zero Licensing Cost: FastAPI, LangChain, FAISS, FastEmbed and the frontend are all open source.",
    "Cloud & Storage Efficiency: Uses AWS S3 presigned URLs for direct document transfers alongside cost-effective PostgreSQL and Redis instances.":
        "Free-Tier Hosting: Render serves the API and the static site; the only cost is the pay-per-use Groq key.",
    "Automated Verification: Local dev and 32-step API test suite (100% pass) run on standard PCs.":
        "Automated Checks: a committed benchmark script scores models and answers on every change.",
}
M[(6, 144)] = S7

S8 = {
    "Dark Glassmorphic UI (Next.js 16 + Tailwind v4) with live API health pill, 1-Click Test User Switcher, and AI Copilot minimizes learning curve.":
        "The chat feels familiar: type or tap chips, watch answers stream word by word, and edit your profile in a side drawer.",
    "Admin governance is streamlined via RolesMaster UUIDs, automated node-cron cleanup jobs, and the Unanswered Questions Triage Queue.":
        "Uploads are guided: a warm summary of the report appears right after upload, and errors are plain-English messages.",
    "All 17 feature modules and 19 REST API route groups (/api/v1) were delivered across structured Agile sprints over a 6-7 week schedule.":
        "The full pipeline \u2014 upload, intake, retrieval, streaming and profile \u2014 was rebuilt and verified in a two-day sprint.",
    "Frontend micro-apps, Novelty AI engines, and Prisma 7 PostgreSQL services were built concurrently to meet all milestones.":
        "Frontend, backend and benchmark tools were built in parallel; test reports and eval scripts ship with the repo.",
    "Verified across First, Second, and Final Reviews with all 32 automated API endpoint assertions passing at 100%.":
        "Verified end-to-end in a real browser across review milestones, with all smoke tests passing.",
}
M[(7, 153)] = S8

S10 = {
    "Client-Side & Dual-Mode Architecture": "Frontend & Chat Interface",
    "Next.js 16 (App Router) & React 19: Powers 17 interactive modules\u2014Dashboard, Space Hub, Nested Threads, AI Copilot, Unanswered Triage, and Admin Panel with TypeScript 5.8.":
        "Vanilla JavaScript + CSS: a zero-build single-page chat with an SSE streaming reader, quick-reply chips, drag-and-drop uploads and a light/dark theme.",
    "Tailwind CSS v4 (Dark Glassmorphism): Implements backdrop-blur glassmorphic panels, Lucide React iconography, and responsive high-contrast developer styling across devices.":
        "Safe Markdown: marked + DOMPurify sanitize every bot reply, so a hostile uploaded document cannot inject scripts into the chat window.",
    "Dual-Mode Runtime & Contexts: AuthContext and api.ts use AbortController (5s/25s timeouts) to connect to Live PostgreSQL APIs or auto-fallback to Offline Demo Mode.":
        "Profile Drawer: sessionStorage keeps the chat per tab; the auto-built profile (age, diet, allergies) can be viewed and edited any time.",
}
M[(9, 171)] = S10

S11 = {
    "Server-Side, Prisma 7 & Storage Layer": "Backend, Data & Model Layer",
    "Express.js 5, TypeScript & Zod 4.3: Asynchronous REST API server (/api/v1) with 20 controllers, strict Zod runtime validation, centralized error handling, and node-cron jobs.":
        "FastAPI + LangChain: async endpoints for upload, streaming chat (SSE), profile and session clear; a two-stage LLM pipeline with strict JSON intake.",
    "PostgreSQL & Prisma 7 ORM: Type-safe relational schema spanning 25+ tables with UUID primary keys, cascade deletes, and Centralized Master Tables (RolesMaster, Tags).":
        "FAISS + FastEmbed: per-session vector stores (1,200-char chunks, 200 overlap) with MMR retrieval; embeddings are computed locally by ONNX.",
    "Redis Cache, AWS S3 & JWT Auth: Combines bcryptjs hashing, Bearer JWTs, Redis (ioredis) token blacklisting, RBAC guards, and AWS S3 presigned URLs for asset distribution.":
        "Groq LPU & Guards: openai/gpt-oss-120b answers; 10 MB and 8-file upload caps, duplicate rejection by hash, sessions evicted after 3 idle hours.",
}
M[(10, 180)] = S11

S12 = {
    "Novelty Features: AI & Problem-Solving Engine": "Novelty: The Intake Engine",
    "AI Technical Copilot & Semantic Vector Search: Context-aware assistant (/assistant) and Cosine Similarity vector matcher rank technical discussions by semantic relevance with excerpt highlights.":
        "Asks Before Answering: a fast JSON call decides whether missing details (age, diet, allergies) would change the answer \u2014 and asks them as quick chips.",
    "Duplicate Detection & Urgency Engine: Pre-post similarity scoring (>75% alert) prevents duplicate threads, while AI auto-classifies domain tags and urgency (Critical, High, Medium, Low).":
        "Personalized RAG: report chunks are woven with the profile and clarified facts, so advice cites the patient\u2019s own values and reference ranges.",
    "Verified Solutions & Unanswered Triage: Emerald-badged Verified Solutions surface expert-attributed answers to the top, while /unanswered routes 0-reply critical issues to domain experts.":
        "Measured Model Choice: a committed benchmark scored four Groq models on JSON reliability and answer quality \u2014 gpt-oss-120b won (5/5 valid intake).",
}
M[(11, 189)] = S12

S13 = {
    "Decoupled Dual-Mode Flow": "Two-Stage Service Flow",
    "Divided into Next.js 16 + React 19": "Browser (JS + SSE) talks to a",
    "Client, Express 5 + TypeScript API,": "FastAPI backend that runs intake",
    "and Prisma 7 PostgreSQL + Redis +": "and answer LLM calls, FAISS search,",
    "AWS S3 Persistence Layer.": "and an in-memory session store.",
    "Requests send Bearer JWT & x-user-id": "Embeddings are computed locally",
    "guarded by 5s/25s AbortController": "by FastEmbed ONNX; report text",
    "timeouts with automatic fallback": "reaches only the Groq model.",
    "to Offline Demo Mode (demo-data.ts).": "Sessions evicted after 3 hours;",
    "Integrates AI Copilot, Semantic": "answers stream token by token",
    "Vector Search & Urgency Triage.": "with a fixed medical disclaimer.",
}
M[(12, 198)] = S13

S14 = {
    "Master Tables & RBAC Schema": "Session & Profile Design",
    "Prisma 7 schema spans 25+ tables": "Each session keeps its own",
    "with UUID keys & cascade rules.": "vector store, file list and",
    "RolesMaster maps fixed UUIDs for": "profile: age, sex, diet,",
    "Admin, Moderator, and Member.": "medicines, allergies, symptoms.",
    "Controllers auto-convert role": "Intake updates the profile",
    "names and enforce RBAC via": "after every turn; answers",
    "verifyAdminAccess guards.": "reuse it without re-asking.",
    "Cron jobs clean expired tokens": "Memory-only storage is",
    "every 15m and polls every 5m.": "evicted after 3 hours idle.",
}
M[(13, 208)] = S14

S15 = {
    "Drafting Research Paper: Focus on AI semantic de-duplication, urgency triage, and resilient dual-mode architectures in decoupled developer platforms (Completed).":
        "Drafting Research Paper: focus on LLM intake engines that ask before answering, personalized RAG over medical reports, and local ONNX embeddings (Completed).",
    "System Verification & Submission: All 17 modules verified via 32 automated API assertions (100% pass rate); paper submission set for 15th Nov 2026.":
        "System Verification & Submission: upload, clarify-chip and streaming flows verified in a real browser; benchmark scored 4 models; submission set for 15th Nov 2026.",
}
M[(14, 219)] = S15

S16 = {
    '[1] S. K. Singh and P. Kumar, "Design and implementation of a decoupled web architecture using React and Node.js," IEEE International Conference on Web Services (ICWS), 2022.':
        '[1] V. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Advances in Neural Information Processing Systems (NeurIPS), 2020.',
    '[2] R. Sharma, "Optimizing relational databases with Centralized Master Tables for dynamic web platforms," IEEE Transactions on Data Engineering, vol. 15, no. 2, pp. 112-120, 2021.':
        '[2] V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," EMNLP, pp. 6769-6781, 2020.',
    '[3] M. Al-Sharafi et al., "The effect of cloud computing and Master Data Management on software efficiency," IEEE Access, vol. 10, pp. 45012-45025, 2022.':
        '[3] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, pp. 3982-3992, 2019.',
    '[4] J. Lee and A. Doe, "Automated deployment of containerized community platforms using AWS and Docker," IEEE Cloud Computing, vol. 9, no. 4, pp. 22-30, 2023.':
        '[4] OpenAI, "gpt-oss: Open-Weight Reasoning Models," OpenAI Technical Report, 2025.',
    '[5] A. Vaskuri, H. Baumgartner, P. K\u00e4rh\u00e4, G. Andor, and E. Ikonen, "Modeling architectural scalability in modern decoupled applications," Journal of Applied Software Engineering, vol. 118, no. 20, pp. 203103-1\u2013203103-7, 2022.':
        '[5] J. Johnson, M. Douze, and H. J\u00e9gou, "Billion-scale similarity search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535-547, 2021.',
}
M[(15, 227)] = S16

TABLE = {  # (slide_idx, row, col) -> new text (header row + Sl.No. column kept)
    (1, 1, 1): 'V. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020.',
    (1, 1, 2): 'Proposed RAG: retrieve supporting passages first, then generate an answer grounded only in them.',
    (1, 1, 3): 'Guided our report-grounded answering: Dr. Asha quotes only the uploaded report values.',
    (1, 2, 1): 'V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," EMNLP 2020.',
    (1, 2, 2): 'Dense embeddings retrieve passages by meaning, not keywords, and stay fast at scale.',
    (1, 2, 3): 'Validated FAISS + local FastEmbed chunks for finding the right lab values quickly.',
    (2, 1, 1): 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP 2019.',
    (2, 1, 2): 'Compact sentence embeddings let small models run quickly on ordinary CPUs.',
    (2, 1, 3): 'Guided our choice of a small local bge embedding model via FastEmbed (ONNX).',
    (2, 2, 1): 'OpenAI, "gpt-oss: Open-Weight Reasoning Models," OpenAI Technical Report, 2025.',
    (2, 2, 2): 'Open-weight models with adjustable reasoning effort, served on fast Groq hardware.',
    (2, 2, 3): 'Guided the model switch: our benchmark picked gpt-oss-120b (5/5 valid JSON intake).',
}

# ---------------------------------------------------------------- run

shutil.copyfile(SRC, OUT)
prs = Presentation(OUT)

for (si, sid), mapping in M.items():
    slide = prs.slides[si]
    hit = False
    for sh in iter_frames(slide.shapes):
        if sh.shape_id == sid:
            apply_map(sh.text_frame, mapping, f"S{si+1}#{sid}")
            hit = True
    if not hit:
        print(f"!! shape {sid} not found on slide {si+1}")

for (si, r, c), new in TABLE.items():
    for sh in prs.slides[si].shapes:
        if sh.has_table:
            cell = sh.table.rows[r].cells[c]
            old = _norm(cell.text)
            replace_in_paragraph(cell.text_frame.paragraphs[0], new)
            for extra in cell.text_frame.paragraphs[1:]:
                for run in list(extra.runs):
                    run._r.getparent().remove(run._r)
            del old

swap_picture(prs, 8, 166, flowchart(), 11.43, 6.02)
swap_picture(prs, 12, 203, architecture(), 5.22, 4.87)
swap_picture(prs, 13, 213, session_design(), 6.39, 5.18)

prs.save(OUT)
print("SAVED", OUT)

# ---------------------------------------------------------------- verify
prs2 = Presentation(OUT)
bad = ["Nexus", "Next.js", "Prisma", "Express", "React 19", "MAGESH", "COMMUNITY PLATFORM",
       "Redis", "AWS S3", "RolesMaster", "Tailwind", "Zod", "node-cron", "PostgreSQL", "Copilot"]
found = []
def walk(shapes, si):
    for sh in shapes:
        if sh.shape_type == 6:
            walk(sh.shapes, si); continue
        if sh.has_text_frame:
            t = sh.text_frame.text
            for w in bad:
                if w.lower() in t.lower():
                    found.append(f"S{si+1}: {w!r} in {t[:60]!r}")
        if sh.has_table:
            for row in sh.table.rows:
                for c in row.cells:
                    for w in bad:
                        if w.lower() in c.text.lower():
                            found.append(f"S{si+1} table: {w!r} in {c.text[:60]!r}")
for i, s in enumerate(prs2.slides):
    walk(s.shapes, i)
print("leftover-check:", "CLEAN" if not found else f"{len(found)} hits")
for f in found:
    print("  -", f)
for i, s in enumerate(prs2.slides):
    for sh in s.shapes:
        if sh.shape_type == 13:
            rId = sh._element.blip_rId
            part = s.part.related_part(rId)
            assert len(part.blob) > 1000, f"S{i+1} broken image"
print("image rels: OK |", len(prs2.slides.__iter__().__length_hint__() if hasattr(prs2.slides, '__length_hint__') else list(prs2.slides)), "slides")
