/* Dr. Asha project deck — pptxgenjs build script. Output: docs/Dr_Asha_Project_Deck.pptx */
const pptxgen = require("pptxgenjs");
const path = require("path");

const ASSETS = path.join(__dirname, "..", "docs", "deck_assets");

// ---- Palette (medical teal, from the app itself) ----
const BG = "F7FAFA";        // light content surface
const BG_DARK = "0B1E22";   // dark cover/closing
const PRIMARY = "0D9488";   // app teal
const PRIMARY_DK = "0F766E";
const DEEP = "115E59";
const TINT = "E6F4F2";      // light teal tint
const TINT2 = "EDF2F2";     // neutral tint (generic card)
const ACCENT = "D97706";    // amber, sparing
const TEXT = "1A2E32";
const MUTED = "5B6B70";
const FAINT = "9DB4B1";     // muted on dark
const LIGHT = "CFE8E5";     // body on dark
const HAIR = "D8E6E4";

const F = "Segoe UI";
const W = 13.33, H = 7.5, M = 0.5;

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "Diwakar R";
p.title = "Dr. Asha — Personalized Medical Report Chatbot";

// ---- Helpers ----
const shadow = () => ({ type: "outer", color: "0B1E22", blur: 7, offset: 2, angle: 60, opacity: 0.18 });

const LOGO = path.join(ASSETS, "logo_512.png");

function badge(s, x, y, size) { // app logo motif: teal badge with white ECG pulse
  s.addImage({ path: LOGO, x, y, w: size, h: size });
}

function header(s, kicker, title, num) {
  s.background = { color: BG };
  s.addText(kicker, { x: M, y: 0.42, w: 11, h: 0.32, fontFace: F, fontSize: 12.5, bold: true, color: PRIMARY, charSpacing: 3, margin: 0 });
  s.addText(title, { x: M, y: 0.74, w: W - 2 * M, h: 0.72, fontFace: F, fontSize: 30, bold: true, color: TEXT, margin: 0 });
  if (num) s.addText(num, { x: W - 1.0, y: 7.02, w: 0.5, h: 0.3, fontFace: F, fontSize: 12, color: FAINT, align: "right", margin: 0 });
}

function srcNote(s, txt) {
  s.addText(txt, { x: M, y: 7.02, w: 9.5, h: 0.3, fontFace: F, fontSize: 12, color: MUTED, margin: 0 });
}

function row(s, x, y, w, lead, desc, leadColor) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.07, w: 0.16, h: 0.16, fill: { color: leadColor || PRIMARY }, rectRadius: 0.04, line: { type: "none" } });
  s.addText(lead, { x: x + 0.34, y, w: w - 0.34, h: 0.32, fontFace: F, fontSize: 16.5, bold: true, color: TEXT, margin: 0 });
  s.addText(desc, { x: x + 0.34, y: y + 0.34, w: w - 0.34, h: 0.62, fontFace: F, fontSize: 13, color: MUTED, margin: 0 });
}

function arrow(s, x, y, w, h, label) {
  s.addShape(p.shapes.LINE, { x, y, w, h, line: { color: PRIMARY, width: 2.2, endArrowType: "triangle" } });
  if (label) s.addText(label, { x: x - 0.25 + w / 2 - 0.75, y: y + h / 2 - 0.34, w: 2.0, h: 0.26, fontFace: F, fontSize: 11.5, italic: true, color: MUTED, align: "center", margin: 0 });
}

// ================= S1 · Cover =================
let s = p.addSlide();
s.background = { color: BG_DARK };
badge(s, 0.95, 1.55, 1.15);
s.addText("Dr. Asha", { x: 0.9, y: 2.95, w: 11.5, h: 1.15, fontFace: F, fontSize: 64, bold: true, color: "FFFFFF", margin: 0 });
s.addText("A medical-report chatbot that asks before it answers", { x: 0.95, y: 4.15, w: 11, h: 0.5, fontFace: F, fontSize: 21, color: LIGHT, margin: 0 });
s.addText("FastAPI  ·  Groq openai/gpt-oss-120b  ·  FAISS + local FastEmbed  ·  SSE streaming  ·  Render", { x: 0.95, y: 4.78, w: 11.5, h: 0.35, fontFace: F, fontSize: 13.5, color: FAINT, margin: 0 });
s.addText([
  { text: "Project walkthrough  ", options: { color: FAINT } },
  { text: "·  live demo: document-chatbot-kambaa.vercel.app", options: { color: "5EEAD4" } },
], { x: 0.95, y: 6.55, w: 11, h: 0.35, fontFace: F, fontSize: 13.5, margin: 0 });
s.addNotes("Hook: most health chatbots give textbook answers. Dr. Asha reads YOUR report and asks about YOU first. Built end-to-end: backend, intake engine, UI, model benchmark, deployment.");

// ================= S2 · Problem =================
s = p.addSlide();
header(s, "THE PROBLEM", "Generic health bots answer everyone the same way", "02");
row(s, M, 2.0, 5.5, "Textbook answers", "Average-patient advice that ignores age, meds, diet and history.");
row(s, M, 3.35, 5.5, "No memory of you", "Every message starts from zero — you re-explain yourself forever.");
row(s, M, 4.7, 5.5, "One-shot guesses", "Never asks the two details that would change the whole answer.");
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.55, y: 1.95, w: 6.28, h: 1.95, fill: { color: TINT2 }, rectRadius: 0.12, line: { type: "none" } });
s.addText("GENERIC BOT", { x: 6.9, y: 2.2, w: 5.6, h: 0.3, fontFace: F, fontSize: 12, bold: true, color: MUTED, charSpacing: 2, margin: 0 });
s.addText("\u201CTiredness has many causes. It is advisable to consult a physician.\u201D", { x: 6.9, y: 2.6, w: 5.6, h: 1.0, fontFace: F, fontSize: 15, italic: true, color: MUTED, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.55, y: 4.25, w: 6.28, h: 2.35, fill: { color: TINT }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addText("DR. ASHA", { x: 6.9, y: 4.5, w: 5.6, h: 0.3, fontFace: F, fontSize: 12, bold: true, color: PRIMARY_DK, charSpacing: 2, margin: 0 });
s.addText("\u201CBecause you\u2019re 45, on metformin, and your HbA1c is 8.4%, this fatigue may also trace back to the anemia in your report (Hb 11.8). Three things to ask your doctor\u2026\u201D", { x: 6.9, y: 4.9, w: 5.6, h: 1.55, fontFace: F, fontSize: 15, color: TEXT, margin: 0 });
s.addNotes("The gap is personalization. Same question, two answers — one is forgettable, one is about the person sitting there.");

// ================= S3 · Pipeline =================
s = p.addSlide();
header(s, "HOW IT THINKS", "Every question runs a two-stage pipeline", "03");
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 3.35, w: 2.05, h: 1.15, fill: { color: "FFFFFF" }, rectRadius: 0.1, line: { color: HAIR, width: 1 }, shadow: shadow() });
s.addText("Your question", { x: M, y: 3.35, w: 2.05, h: 1.15, fontFace: F, fontSize: 15, bold: true, color: TEXT, align: "center", valign: "middle", margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 3.3, y: 2.75, w: 3.35, h: 2.35, fill: { color: PRIMARY }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addText([
  { text: "STAGE 1 \u00B7 INTAKE", options: { fontSize: 13, bold: true, color: "FFFFFF", charSpacing: 2, breakLine: true } },
  { text: "One fast JSON call: extracts durable facts into a session profile, then decides \u2014 answer now, or ask first?", options: { fontSize: 13, color: "D9F3F0" } },
], { x: 3.55, y: 2.95, w: 2.9, h: 2.0, fontFace: F, paraSpaceAfter: 8, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 7.55, y: 1.55, w: 5.28, h: 1.5, fill: { color: "FFFFFF" }, rectRadius: 0.12, line: { color: HAIR, width: 1 }, shadow: shadow() });
s.addText([
  { text: "QUICK-REPLY CHIPS", options: { fontSize: 12.5, bold: true, color: ACCENT, charSpacing: 2, breakLine: true } },
  { text: "\u201CVegetarian or non-veg?\u201D \u00B7 \u201CAny allergies?\u201D \u00B7 \u201CSince when?\u201D \u2014 answered in two clicks", options: { fontSize: 13, color: TEXT } },
], { x: 7.85, y: 1.75, w: 4.7, h: 1.15, fontFace: F, paraSpaceAfter: 6, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 7.55, y: 4.6, w: 5.28, h: 1.75, fill: { color: DEEP }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addText([
  { text: "STAGE 2 \u00B7 ANSWER", options: { fontSize: 12.5, bold: true, color: "5EEAD4", charSpacing: 2, breakLine: true } },
  { text: "Retrieval over your reports + your profile + what you clarified \u2192 streamed live in the Dr. Asha persona", options: { fontSize: 13, color: "FFFFFF" } },
], { x: 7.85, y: 4.8, w: 4.7, h: 1.4, fontFace: F, paraSpaceAfter: 6, margin: 0 });
arrow(s, 2.6, 3.93, 0.62, 0);
arrow(s, 6.7, 3.3, 0.8, -1.1);
arrow(s, 6.7, 4.35, 0.8, 1.0);
s.addText("ask", { x: 6.62, y: 2.85, w: 0.9, h: 0.26, fontFace: F, fontSize: 11.5, italic: true, color: MUTED, align: "center", margin: 0 });
s.addText("answer", { x: 6.62, y: 4.5, w: 0.9, h: 0.26, fontFace: F, fontSize: 11.5, italic: true, color: MUTED, align: "center", margin: 0 });
s.addText("Profile updates silently every turn \u2014 age, diet, medications, allergies, preferences.", { x: 3.3, y: 5.55, w: 3.6, h: 0.85, fontFace: F, fontSize: 12.5, italic: true, color: MUTED, margin: 0 });
s.addNotes("Stage 1 is the differentiator: a cheap JSON call that decides whether missing details would materially change the answer. Round-capped at 2 so it never loops.");

// ================= S4 · Personalization in action =================
s = p.addSlide();
header(s, "PERSONALIZATION", "It asks, then it tailors \u2014 a real session", "04");
const steps = [
  ["Ask naturally", "\u201CWhat can I eat for lunch today?\u201D"],
  ["It asks what matters", "Diet preference \u00B7 allergies \u00B7 cuisine \u2014 as clickable chips"],
  ["The answer is yours", "Peanut-free Indian vegetarian plan, tied to the anemia and glucose values in the report"],
];
steps.forEach(([lead, desc], i) => {
  const y = 2.0 + i * 1.5;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y, w: 0.5, h: 0.5, fill: { color: PRIMARY }, rectRadius: 0.12, line: { type: "none" } });
  s.addText(String(i + 1), { x: M, y, w: 0.5, h: 0.5, fontFace: F, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText(lead, { x: 1.2, y: y - 0.02, w: 4.9, h: 0.35, fontFace: F, fontSize: 16.5, bold: true, color: TEXT, margin: 0 });
  s.addText(desc, { x: 1.2, y: y + 0.33, w: 4.9, h: 0.95, fontFace: F, fontSize: 13.5, color: MUTED, margin: 0 });
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.45, y: 1.7, w: 6.4, h: 4.35, fill: { color: "FFFFFF" }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addImage({ path: path.join(ASSETS, "chips_dark.png"), x: 6.6, y: 1.85, w: 6.1, h: 3.43 });
s.addText("Live session \u2014 the bot collected diet and cuisine before recommending anything.", { x: 6.6, y: 5.38, w: 6.1, h: 0.5, fontFace: F, fontSize: 12, color: MUTED, margin: 0 });
s.addNotes("This is the exact flow the user asked for: input first, personalized output second. Screenshot from the running app.");

// ================= S5 · Report intelligence =================
s = p.addSlide();
header(s, "REPORT INTELLIGENCE", "Upload a report \u2014 it reads it before you ask", "05");
const flow = [
  ["Parse anything", "PDF \u00B7 DOCX \u00B7 TXT \u00B7 CSV \u00B7 XLSX \u2014 tables included"],
  ["Chunk smartly", "1,200 characters, 200 overlap, recursive splitting"],
  ["Embed locally", "FastEmbed ONNX \u2014 no API key, no third party"],
  ["Index per session", "FAISS with MMR retrieval, isolated per chat"],
  ["Orient instantly", "A warm plain-language summary the moment it lands"],
];
flow.forEach(([lead, desc], i) => {
  const y = 1.85 + i * 0.98;
  const last = i === flow.length - 1;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: y + 0.05, w: 0.34, h: 0.34, fill: { color: last ? ACCENT : PRIMARY }, rectRadius: 0.08, line: { type: "none" } });
  s.addText(String(i + 1), { x: M, y: y + 0.05, w: 0.34, h: 0.34, fontFace: F, fontSize: 13, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText([{ text: lead + "  \u2014  ", options: { bold: true, color: last ? ACCENT : TEXT } }, { text: desc, options: { color: MUTED } }],
    { x: 1.05, y, w: 6.3, h: 0.85, fontFace: F, fontSize: 14.5, margin: 0 });
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 7.75, y: 1.85, w: 5.08, h: 2.0, fill: { color: TINT }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addText("414 tok/s", { x: 7.75, y: 2.1, w: 5.08, h: 1.0, fontFace: F, fontSize: 54, bold: true, color: PRIMARY_DK, align: "center", margin: 0 });
s.addText("answer streaming on gpt-oss-120b \u2014 first words in ~1s", { x: 7.9, y: 3.2, w: 4.78, h: 0.5, fontFace: F, fontSize: 13, color: MUTED, align: "center", margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 7.75, y: 4.1, w: 2.44, h: 1.7, fill: { color: "FFFFFF" }, rectRadius: 0.12, line: { color: HAIR, width: 1 } });
s.addText("5", { x: 7.75, y: 4.3, w: 2.44, h: 0.7, fontFace: F, fontSize: 40, bold: true, color: TEXT, align: "center", margin: 0 });
s.addText("formats parsed", { x: 7.75, y: 5.1, w: 2.44, h: 0.4, fontFace: F, fontSize: 12.5, color: MUTED, align: "center", margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 10.39, y: 4.1, w: 2.44, h: 1.7, fill: { color: "FFFFFF" }, rectRadius: 0.12, line: { color: HAIR, width: 1 } });
s.addText("8", { x: 10.39, y: 4.3, w: 2.44, h: 0.7, fontFace: F, fontSize: 40, bold: true, color: TEXT, align: "center", margin: 0 });
s.addText("reports per session", { x: 10.39, y: 5.1, w: 2.44, h: 0.4, fontFace: F, fontSize: 12.5, color: MUTED, align: "center", margin: 0 });
s.addNotes("Numbers measured during live testing: ~414 tokens/s streaming, 5 file formats, 8-report session cap. Reports accumulate — question 2 can cite report 1.");

// ================= S6 · Model benchmark =================
s = p.addSlide();
header(s, "MODEL SELECTION", "We benchmarked four models \u2014 then picked the smart one", "06");
s.addChart(p.charts.BAR, [
  { name: "Valid intake JSON (%)", labels: ["gpt-oss-20b", "gpt-oss-120b", "safeguard-20b", "qwen3.8-27b"], values: [60, 100, 60, 100] },
  { name: "Correct ask/answer call (%)", labels: ["gpt-oss-20b", "gpt-oss-120b", "safeguard-20b", "qwen3.8-27b"], values: [40, 80, 40, 80] },
], {
  x: M, y: 1.8, w: 7.9, h: 4.8, barDir: "col", barGapWidthPct: 90,
  chartColors: [PRIMARY, "5EEAD4"],
  chartArea: { fill: { color: BG } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  valAxisMaxVal: 100, valAxisMajorUnit: 25,
  valGridLine: { color: "E2ECEA", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 11,
  showLegend: true, legendPos: "b", legendColor: MUTED, legendFontSize: 12,
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.75, y: 2.0, w: 4.08, h: 3.4, fill: { color: TINT }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addText([
  { text: "Why we switched", options: { fontSize: 16, bold: true, color: TEXT, breakLine: true } },
  { text: "gpt-oss-20b writes beautifully (640 tok/s) \u2014 but Groq\u2019s strict JSON validator rejects its intake output ~40% of the time, silently breaking clarification.", options: { fontSize: 13, color: MUTED, breakLine: true } },
  { text: "gpt-oss-120b: 5/5 valid JSON, equally fast in practice. It\u2019s the default.", options: { fontSize: 13, bold: true, color: PRIMARY_DK } },
], { x: 9.05, y: 2.25, w: 3.5, h: 3.0, fontFace: F, paraSpaceAfter: 10, margin: 0 });
srcNote(s, "Source: tools/eval_groq_models.py \u2014 5 intake scenarios + scored RAG answers per model, Groq API, Nov 2026.");
s.addNotes("Honest engineering: the user's first pick (20b) failed the pipeline's JSON contract. The benchmark script ships with the repo so the choice is reproducible.");

// ================= S7 · Safety =================
s = p.addSlide();
header(s, "SAFETY", "A doctor\u2019s bedside manner, with guardrails", "07");
const guards = [
  ["Grounded, never invented", "Quotes exact values with units and reference ranges from your report."],
  ["Never diagnoses", "Explains associations, then points to your doctor for any conclusion."],
  ["Red flags escalate", "Chest pain, breathlessness, sudden weakness \u2192 seek urgent care now."],
  ["Speaks human", "Every medical term defined in plain words on first use."],
  ["Always discloses", "The AI-not-a-doctor disclaimer closes every single answer."],
];
guards.forEach(([lead, desc], i) => row(s, M, 1.95 + i * 0.99, 7.1, lead, desc));
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 2.2, w: 4.6, h: 2.5, fill: { color: PRIMARY }, rectRadius: 0.14, line: { type: "none" }, shadow: shadow() });
s.addText("6/6", { x: 8.2, y: 2.5, w: 4.6, h: 1.1, fontFace: F, fontSize: 60, bold: true, color: "FFFFFF", align: "center", margin: 0 });
s.addText("automated quality checks passed by the default model", { x: 8.45, y: 3.75, w: 4.1, h: 0.7, fontFace: F, fontSize: 13.5, color: "D9F3F0", align: "center", margin: 0 });
s.addText("checks: uses your name \u00B7 quotes real values \u00B7 no definitive diagnosis \u00B7 disclaimer present \u00B7 next steps \u00B7 personalized opening", { x: 8.2, y: 5.0, w: 4.6, h: 0.9, fontFace: F, fontSize: 11.5, color: MUTED, margin: 0 });
s.addNotes("The 6/6 is the scored answer task in the benchmark script — automatic checks, not vibes.");

// ================= S8 · Architecture =================
s = p.addSlide();
header(s, "ARCHITECTURE", "Modern stack, honest about every hop", "08");
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 2.5, w: 3.3, h: 1.9, fill: { color: "FFFFFF" }, rectRadius: 0.12, line: { color: HAIR, width: 1 }, shadow: shadow() });
s.addText([
  { text: "BROWSER", options: { fontSize: 12.5, bold: true, color: PRIMARY_DK, charSpacing: 2, breakLine: true } },
  { text: "Vanilla JS + CSS \u00B7 SSE stream reader \u00B7 quick-reply chips \u00B7 DOMPurify-safe markdown \u00B7 profile drawer", options: { fontSize: 12.5, color: TEXT } },
], { x: 0.72, y: 2.7, w: 2.9, h: 1.6, fontFace: F, paraSpaceAfter: 7, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 4.95, y: 2.2, w: 3.55, h: 2.5, fill: { color: PRIMARY }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addText([
  { text: "FASTAPI BACKEND", options: { fontSize: 12.5, bold: true, color: "FFFFFF", charSpacing: 2, breakLine: true } },
  { text: "Intake + answer LLM calls \u00B7 FAISS vector store (MMR) \u00B7 in-memory sessions with TTL \u00B7 SSE StreamingResponse", options: { fontSize: 12.5, color: "D9F3F0" } },
], { x: 5.2, y: 2.4, w: 3.1, h: 2.2, fontFace: F, paraSpaceAfter: 7, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 9.7, y: 2.5, w: 3.13, h: 1.9, fill: { color: BG_DARK }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addText([
  { text: "GROQ LPU", options: { fontSize: 12.5, bold: true, color: "5EEAD4", charSpacing: 2, breakLine: true } },
  { text: "openai/gpt-oss-120b \u2014 the only hop report text ever takes", options: { fontSize: 12.5, color: "FFFFFF" } },
], { x: 9.95, y: 2.7, w: 2.65, h: 1.6, fontFace: F, paraSpaceAfter: 7, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 4.95, y: 5.35, w: 3.55, h: 1.15, fill: { color: TINT }, rectRadius: 0.12, line: { type: "none" } });
s.addText([
  { text: "FASTEMBED (ONNX, local)", options: { fontSize: 12, bold: true, color: PRIMARY_DK, charSpacing: 1, breakLine: true } },
  { text: "embeddings computed on your own machine", options: { fontSize: 12, color: MUTED } },
], { x: 5.2, y: 5.5, w: 3.1, h: 0.9, fontFace: F, paraSpaceAfter: 5, margin: 0 });
s.addShape(p.shapes.LINE, { x: 3.8, y: 3.45, w: 1.15, h: 0, line: { color: PRIMARY, width: 2.2, beginArrowType: "triangle", endArrowType: "triangle" } });
s.addText("SSE", { x: 3.85, y: 3.05, w: 1.05, h: 0.3, fontFace: F, fontSize: 11.5, italic: true, color: MUTED, align: "center", margin: 0 });
s.addShape(p.shapes.LINE, { x: 8.5, y: 3.45, w: 1.2, h: 0, line: { color: PRIMARY, width: 2.2, beginArrowType: "triangle", endArrowType: "triangle" } });
s.addText("chat API", { x: 8.5, y: 3.05, w: 1.2, h: 0.3, fontFace: F, fontSize: 11.5, italic: true, color: MUTED, align: "center", margin: 0 });
s.addShape(p.shapes.LINE, { x: 6.72, y: 4.7, w: 0, h: 0.65, line: { color: PRIMARY, width: 2.2, endArrowType: "triangle" } });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 9.7, y: 5.35, w: 3.13, h: 1.15, fill: { color: "FFFFFF" }, rectRadius: 0.12, line: { color: HAIR, width: 1 } });
s.addText([
  { text: "RENDER BLUEPRINT", options: { fontSize: 12, bold: true, color: TEXT, charSpacing: 1, breakLine: true } },
  { text: "one Python service + one static site, infra as code", options: { fontSize: 12, color: MUTED } },
], { x: 9.95, y: 5.5, w: 2.65, h: 0.9, fontFace: F, paraSpaceAfter: 5, margin: 0 });
s.addNotes("Zero build step on the frontend; the backend serves the UI too, so one command runs everything locally.");

// ================= S9 · Privacy =================
s = p.addSlide();
header(s, "PRIVACY & HARDENING", "Medical data deserves restraint", "09");
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 2.0, w: 3.7, h: 3.1, fill: { color: TINT }, rectRadius: 0.14, line: { type: "none" }, shadow: shadow() });
s.addText("0 bytes", { x: M, y: 2.35, w: 3.7, h: 1.05, fontFace: F, fontSize: 54, bold: true, color: PRIMARY_DK, align: "center", margin: 0 });
s.addText("persisted to disk \u2014 reports, profiles and chats live in memory only, evicted after 3 hours idle", { x: 0.75, y: 3.55, w: 3.2, h: 1.3, fontFace: F, fontSize: 13.5, color: TEXT, align: "center", margin: 0 });
const priv = [
  ["Embeddings never leave the machine", "Local ONNX inference \u2014 no embedding API sees your report."],
  ["One egress point, on purpose", "Report text reaches only the Groq LLM endpoint. Nowhere else."],
  ["Uploads are fenced", "10 MB cap \u00B7 8 files per session \u00B7 duplicates rejected by content hash."],
  ["Output is sanitized", "DOMPurify-clean markdown \u2014 a hostile document can\u2019t inject scripts."],
];
priv.forEach(([lead, desc], i) => row(s, 4.75, 2.0 + i * 1.13, 8.08, lead, desc));
s.addNotes("Deliberate trade-off: memory-only sessions keep the demo honest about privacy, at the cost of persistence. Auth + rate limiting are the next hardening step.");

// ================= S10 · Tested end-to-end =================
s = p.addSlide();
header(s, "VERIFICATION", "Proven against complex hospital reports", "10");
const fixtures = [
  ["Metabolic_CBC_Lipid_Panel_Report.docx", "45-year-old with T2DM \u2014 CMP, CBC, lipids, urinalysis, impression"],
  ["Thyroid_VitaminD_Iron_Panel_Report.docx", "32-year-old fatigue workup \u2014 thyroid, vitamin D, iron, hormones"],
  ["Lab_Trends_Workbook.xlsx", "Four sheets of two-year HbA1c, lipid, BP and lifestyle trends"],
];
fixtures.forEach(([lead, desc], i) => row(s, M, 1.95 + i * 1.05, 5.7, lead, desc));
s.addText("All three are synthetic and generated by tools/generate_test_reports.py \u2014 complex, flagged, realistic.", { x: M, y: 5.3, w: 5.7, h: 0.8, fontFace: F, fontSize: 12.5, italic: true, color: MUTED, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.45, y: 1.7, w: 6.4, h: 4.35, fill: { color: "FFFFFF" }, rectRadius: 0.12, line: { type: "none" }, shadow: shadow() });
s.addImage({ path: path.join(ASSETS, "answer_light.png"), x: 6.6, y: 1.85, w: 6.1, h: 3.43 });
s.addText("Browser-verified: upload \u2192 summary \u2192 clarify chips \u2192 streamed, report-grounded answer.", { x: 6.6, y: 5.38, w: 6.1, h: 0.5, fontFace: F, fontSize: 12, color: MUTED, margin: 0 });
s.addNotes("Testing discipline: synthetic fixtures mean no real PHI, and the generator script is committed so anyone can regenerate.");

// ================= S11 · Closing =================
s = p.addSlide();
s.background = { color: BG_DARK };
badge(s, M, 0.9, 0.85);
s.addText("Ask it something personal.", { x: M, y: 2.3, w: 12.3, h: 1.0, fontFace: F, fontSize: 48, bold: true, color: "FFFFFF", margin: 0 });
s.addText([
  { text: "Running now: ", options: { color: LIGHT } },
  { text: "http://127.0.0.1:8000", options: { color: "5EEAD4", bold: true } },
  { text: "    \u00B7    deployed: render.yaml + one GROQ_API_KEY", options: { color: LIGHT } },
], { x: M, y: 3.5, w: 12.3, h: 0.45, fontFace: F, fontSize: 17, margin: 0 });
s.addText("ON THE ROADMAP", { x: M, y: 4.75, w: 6, h: 0.3, fontFace: F, fontSize: 12, bold: true, color: FAINT, charSpacing: 3, margin: 0 });
[["Multi-file drop", "upload all reports at once"], ["Auth + rate limits", "before real patients, not after"], ["Scanned-PDF OCR", "image-only reports welcome"]].forEach(([lead, desc], i) => {
  const x = M + i * 4.2;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 5.15, w: 0.16, h: 0.16, fill: { color: PRIMARY }, rectRadius: 0.04, line: { type: "none" } });
  s.addText(lead, { x: x + 0.32, y: 5.05, w: 3.6, h: 0.32, fontFace: F, fontSize: 15.5, bold: true, color: "FFFFFF", margin: 0 });
  s.addText(desc, { x: x + 0.32, y: 5.38, w: 3.6, h: 0.35, fontFace: F, fontSize: 12.5, color: FAINT, margin: 0 });
});
s.addText("Dr. Asha \u00B7 an AI companion, not a substitute for professional medical advice", { x: M, y: 6.75, w: 12.3, h: 0.35, fontFace: F, fontSize: 12.5, italic: true, color: FAINT, margin: 0 });
s.addNotes("Close on the demo. The roadmap is ranked by what stands between the demo and real patients.");

p.writeFile({ fileName: path.join(__dirname, "..", "docs", "Dr_Asha_Project_Deck.pptx") }).then((f) => console.log("WROTE", f));
