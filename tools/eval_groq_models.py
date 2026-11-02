"""Benchmark candidate Groq models for the Dr. Asha pipeline.

Measures, per model:
  A. INTAKE reliability  — does it return valid, schema-shaped JSON with the
     correct clarify-vs-answer decision across 5 clinical scenarios?
  B. ANSWER quality      — a RAG-style personalized answer scored with
     automatic checks (uses profile facts, quotes real values with units,
     no definitive diagnosis, disclaimer present) + latency + throughput.

Run:  backend/.venv/Scripts/python tools/eval_groq_models.py
Requires GROQ_API_KEY in backend/.env.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "backend" / ".env")

MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.8-27b",
]

client = Groq(api_key=os.environ["GROQ_API_KEY"])

INTAKE_SYSTEM = """You are the INTAKE module of "Dr. Asha", a virtual health assistant.
You receive a PROFILE, PENDING questions, and the user's NEW MESSAGE. Decide:
1. profile_updates: durable personal facts stated in the message (keys: name, age, sex, pregnant,
   conditions[], medications[], allergies[], symptoms[], symptom_duration, preferences[], notes[]).
2. action: "answer" if we can give a materially personalized answer now; "clarify" ONLY if 1-3
   missing details would MATERIALLY change the answer (age/sex for lab ranges, pregnancy, meds,
   symptom duration). Never ask for info already present. Never clarify simple lookups.
3. questions (only when clarifying): [{"id","question","type":"choice"|"text","options":[...]}].
Respond with ONLY JSON: {"profile_updates":{}, "action":"answer|clarify", "preamble":"", "questions":[]}"""


INTAKE_CASES = [
    # (name, profile_json, pending, message, expected_action)
    ("new user vague symptom", "{}", None,
     "I have been feeling really tired lately.", "clarify"),
    ("full profile lab question", '{"age":45,"sex":"male","conditions":["Type 2 diabetes"],"medications":["Metformin"]}', None,
     "My HbA1c is 8.4% — what does that mean for me?", "answer"),
    ("simple lookup", '{"age":32}', None,
     "What does TSH mean in my report?", "answer"),
    ("reply to pending", "{}",
     {"question": "I have been feeling really tired lately.",
      "questions": [{"id": "duration", "question": "How long has the tiredness lasted?",
                     "type": "choice", "options": ["Less than 2 weeks", "2-6 weeks", "Over 2 months"]}],
      "round": 1},
     "It has been about 3 months now, and I also get palpitations.", "answer"),
    ("pregnancy-sensitive", '{"sex":"female"}', None,
     "Is it safe to take iron tablets?", "clarify"),
]


def score_intake(model):
    ok_json = ok_action = 0
    latencies = []
    for name, profile, pending, message, expected in INTAKE_CASES:
        user = (f"PROFILE: {profile}\nPENDING_QUESTIONS: {json.dumps(pending) if pending else 'null'}\n"
                f"USER'S NEW MESSAGE: {message}\nReturn the JSON now.")
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.1, max_completion_tokens=900,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": INTAKE_SYSTEM},
                          {"role": "user", "content": user}],
            )
            dt = time.time() - t0
            latencies.append(dt)
            data = json.loads(r.choices[0].message.content or "{}")
            ok_json += 1
            if data.get("action") == expected:
                ok_action += 1
            else:
                print(f"    [action miss] {name}: expected {expected}, got {data.get('action')}")
        except Exception as exc:
            print(f"    [error] {name}: {str(exc)[:120]}")
    n = len(INTAKE_CASES)
    tps = None
    if latencies:
        return {"valid_json": f"{ok_json}/{n}", "correct_action": f"{ok_action}/{n}",
                "avg_latency_s": round(sum(latencies) / len(latencies), 2)}
    return {"valid_json": f"{ok_json}/{n}", "correct_action": f"{ok_action}/{n}", "avg_latency_s": None}


RAG_CONTEXT = """[from Metabolic_CBC_Lipid_Panel_Report.docx]
HbA1c (NGSP-aligned): 8.4 % [H] (goal < 7.0 for T2DM) | Fasting Blood Glucose: 168 mg/dL [H]
LDL Cholesterol: 158 mg/dL [H] | HDL: 34 mg/dL [L] | Triglycerides: 242 mg/dL [H]
Urine Albumin-Creatinine Ratio: 62 mg/g [H] | eGFR: 78 | Hemoglobin: 11.8 g/dL [L], MCV 74 fL [L]"""

ANSWER_SYSTEM = f"""You are "Dr. Asha", a warm virtual health assistant. Use the profile to personalize.
Never give a definitive diagnosis. End with exactly: "I'm an AI assistant, not a substitute for professional medical advice — please confirm anything important with your doctor or a qualified healthcare professional."
REPORT EXCERPTS:\n{RAG_CONTEXT}"""


def score_answer(model):
    question = "My doctor said my HbA1c is high — what does it mean for me and what should I do?"
    profile = '{"name":"Rajesh","age":45,"sex":"male","conditions":["Type 2 diabetes","Hypertension"],"medications":["Metformin 500mg","Telmisartan 40mg"]}'
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, temperature=0.55,
        messages=[{"role": "system", "content": ANSWER_SYSTEM},
                  {"role": "user", "content": f"ABOUT THIS PERSON: {profile}\n\n{question}"}],
    )
    dt = time.time() - t0
    text = r.choices[0].message.content or ""
    usage = r.usage

    checks = {
        "uses_name": "rajesh" in text.lower(),
        "quotes_value_8.4": "8.4" in text,
        "mentions_ldl_or_glucose": ("168" in text) or ("ldl" in text.lower()),
        "no_definitive_dx": ("you have diabetes" not in text.lower()) and ("you are diabetic" not in text.lower()),
        "disclaimer": "not a substitute for professional medical advice" in text.lower(),
        "actionable_next_steps": any(k in text.lower() for k in ["ask your doctor", "next", "consider", "discuss"]),
    }
    score = sum(checks.values())
    failed = [k for k, v in checks.items() if not v]
    return {
        "checks": f"{score}/{len(checks)}",
        "failed": failed,
        "latency_s": round(dt, 2),
        "tok_s": round(usage.completion_tokens / (dt - getattr(r, 'queue_time', 0) or dt), 1) if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "answer_preview": text[:220].replace("\n", " "),
    }


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for model in (m for m in MODELS if not only or m in only):
        print(f"\n=== {model} ===")
        print("  INTAKE:", json.dumps(score_intake(model)))
        try:
            ans = score_answer(model)
            print(f"  ANSWER: checks={ans['checks']} failed={ans['failed']} "
                  f"latency={ans['latency_s']}s tok/s={ans['tok_s']}")
            print(f"  PREVIEW: {ans['answer_preview']}...")
        except Exception as exc:
            print(f"  ANSWER: failed — {str(exc)[:140]}")


if __name__ == "__main__":
    main()
