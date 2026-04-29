from __future__ import annotations

import os, re, requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("API_KEY")
API_BASE = "https://openai.rc.asu.edu/v1"
MODEL    = "qwen3-30b-a3b-instruct-2507"


def call_llm(prompt, system="You are a helpful assistant. Reply with only the final answer, no explanation.", temperature=0.0, max_tokens=256, retries=2):
    url     = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            print("API error:", r.status_code, r.text)
            return ""
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"Timeout, retrying ({attempt + 1}/{retries})...")
            else:
                print("Request timed out after all retries.")
                return ""
        except Exception as e:
            print("Request failed:", e)
            return ""


# --- answer cleaning ---

def clean_answer(text):
    if not text:
        return ""
    text = text.strip()
    # strip markdown code fences (handles ``` and ```python etc.)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    text = text.strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    for pattern in [r"(?i)verified answer\s*[:\-]\s*(.*)", r"(?i)corrected answer\s*[:\-]\s*(.*)",
                    r"(?i)final answer\s*[:\-]\s*(.*)", r"(?i)solution\s*[:\-]\s*(.*)",
                    r"(?i)answer\s*[:\-]\s*(.*)"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        for l in reversed(lines):
            if not re.match(r"(?i)^step\s*\d+", l):
                return l[:4999]
    return text[:4999]


def extract_mcq_letter(text):
    if not text:
        return ""
    m = re.search(r"\(([A-E])\)", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", text)
    return m.group(1) if m else ""


def normalize_yes_no(text):
    if not text:
        return ""
    low = text.lower().strip()
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"
    if " yes" in f" {low}":
        return "Yes"
    if " no" in f" {low}":
        return "No"
    return text.strip()


def normalize_for_vote(answer, qtype):
    answer = clean_answer(answer)
    if not answer:
        return ""
    if qtype == "mcq":
        letter = extract_mcq_letter(answer)
        return letter if letter else answer
    if qtype == "yesno":
        return normalize_yes_no(answer)
    if qtype == "math":
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", answer)
        return nums[-1].replace(",", "") if nums else ""
    return answer.strip()


def majority_vote(answers, qtype="general"):
    cleaned = [normalize_for_vote(a, qtype) for a in answers if a]
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


# question classification

def classify(q):
    text  = q.lower()
    start = q[:400].lower()

    # coding questions
    code_signals = ["the function should output", "you should write self-contained code",
                    "def task_func", "complete the following code", "```python"]
    if any(s in text for s in code_signals):
        return "code"

    # planning questions
    plan_signals = ["[plan]", "[plan end]", "[statement]", "my plan is as follows",
                    "here are the actions i can do"]
    if any(s in text for s in plan_signals):
        return "plan"

    # MCQ
    has_yesno_options = "options:\n- yes\n- no" in text or "options:\n(a) yes\n(b) no" in text
    if not has_yesno_options:
        if "options:" in start or re.search(r"\([a-e]\)", start) or re.search(r"(?:^|\s)[a-d]\.\s", start):
            return "mcq"

    # long-context questions
    has_long_context = "context:" in text or len(text) > 600
    if has_long_context:
        return "general"

    yesno_starts = ("does ", "do ", "did ", "is ", "are ", "can ", "would ", "could ", "should ", "was ", "were ", "will ")
    yesno_phrases = ("yes or no", "plausible?", "is the following sentence plausible")
    if has_yesno_options or any(p in text for p in yesno_phrases) or any(text.startswith(s) for s in yesno_starts):
        return "yesno"

    math_words = [
        "calculate", "how many", "how much", "total", "percent", "%",
        "commission", "cost", "earned", "saved", "profit", "price", "per",
        "twice", "double", "triple", "sum", "difference", "product", "times",
    ]
    if any(w in text for w in math_words) or re.search(r'\$\d+', text):
        return "math"

    return "general"


# # technique 8: translation

def is_non_english(text):
    return sum(1 for c in text if ord(c) > 127) / max(len(text), 1) > 0.3


def translate(text):
    result = call_llm(
        "Translate the following to English. Keep all numbers and names exactly as they are. Return only the translation.\n\n" + text,
        max_tokens=512,
    )
    return result.strip() if result.strip() else text



# technique 1: zero-shot direct
def prompt_direct(q, qtype):
    extra = ""
    if qtype == "math":
        extra = (
            "\n- Convert all time units first (e.g. 3 hours = 180 minutes)."
            "\n- If asking for a total, sum every relevant value."
            "\n- If asking for remaining or missing, subtract from the target."
            "\n- Make sure you answered what was asked, not an intermediate value."
        )
    if qtype == "general":
        extra = "\n- Keep your answer short: a single name, phrase, or number when possible."
    return f"""Answer the question. Read carefully and answer exactly what is being asked.

Rules:
- Return only the final answer, no explanation.
- For multiple choice, return the option letter and option.
- For yes/no, return only Yes or No.
- Do not copy sentences from the question as your answer.{extra}

Question type: {qtype}
Question: {q}"""


# technique 2: chain-of-thought
def prompt_cot(q, qtype):
    if qtype == "math":
        return f"""Solve this step by step. Write out every calculation.
Convert all units first. After finishing, re-read the question to confirm your answer matches what was asked.

On the very last line write: Final Answer: <number>

Question: {q}"""
    return f"""Solve carefully step by step, then return only the final answer.
For multiple choice, return only the letter. For yes/no, return only Yes or No.
Keep the final answer short.

Question type: {qtype}
Question: {q}"""


# technique 3: decomposition
def prompt_decompose(q, qtype):
    return f"""Break this into parts. For each named person, item, or category write its value on its own line.
Then combine them to get the final answer. Re-read the question to confirm you answered what was asked.

On the very last line write: Final Answer: <number>

Question type: {qtype}
Question: {q}"""


# technique 4: self-refine
def prompt_refine(q, draft, qtype):
    extra = ""
    if qtype == "math":
        extra = "\nCheck: did you answer what was actually asked? Did you convert units? Did you include all parts?"
    return f"""Check the previous answer and correct it if wrong.{extra}

Question type: {qtype}
Question: {q}
Previous answer: {draft}

Return only the final corrected answer. For multiple choice, return only the letter. For yes/no, return only Yes or No."""


# technique 5: option elimination (MCQ only)
def prompt_eliminate(q):
    return f"""Use option elimination. Rule out wrong answers one by one, then return only the correct option letter.

Question: {q}"""


# technique 6: math verification
def prompt_verify(q, draft):
    return f"""Someone answered this math problem with "{draft}".
Solve it yourself from scratch to verify. Show your work.
Make sure you answer what was asked (total, remaining, missing), not an intermediate step.

Question: {q}

On the last line write: Verified Answer: <number>"""


def prompt_code(q):
    return f"""Write the requested code. Return only the code, no explanation or anyting else.

Question: {q}"""



def prompt_plan(q):
    return f"""Read the problem carefully. Produce only the next plan that achieves the stated goal.
Use the exact action format shown in the examples. Output only the plan and nothing else.

Question: {q}"""


# technique 7: self-consistency (called with temperature > 0, majority voted externally)




def answer_question(q):
    q = q.strip()

    # technique 8: translate if non-English
    if is_non_english(q):
        q = translate(q)

    qtype = classify(q)

    # coding questions
    if qtype == "code":
        code_system = "You are a coding assistant. Return only the code requested, no explanation, comments or anything else."
        result = call_llm(prompt_code(q), system=code_system, max_tokens=1024)
        # strip fences but otherwise return the code as-is
        result = re.sub(r"```[a-zA-Z]*\n?", "", result)
        result = re.sub(r"\n?```", "", result).strip()
        return result[:4999]

    # planning questions: ask for a multi-line plan, return as-is without line extraction
    if qtype == "plan":
        plan_system = "You are a planning assistant. Output only the requested plan in the exact format shown. No explanation."
        result = call_llm(prompt_plan(q), system=plan_system, max_tokens=2048)
        return result.strip()[:4999]

    answers = []
    tokens  = 512 if qtype in {"math", "general"} else 256

    # technique 1: zero-shot direct
    a_direct = clean_answer(call_llm(prompt_direct(q, qtype)))
    answers.append(a_direct)

    # technique 2: chain-of-thought
    a_cot = clean_answer(call_llm(prompt_cot(q, qtype), max_tokens=tokens))
    answers.append(a_cot)

    # technique 3: decomposition for math
    if qtype == "math":
        a_decomp = clean_answer(call_llm(prompt_decompose(q, qtype), max_tokens=512))
        answers.append(a_decomp)

    draft    = majority_vote(answers, qtype) or a_direct
    disagree = normalize_for_vote(a_direct, qtype) != normalize_for_vote(a_cot, qtype)

    # technique 4: self-refine
    if disagree:
        a_refine = clean_answer(call_llm(prompt_refine(q, draft, qtype), max_tokens=tokens))
        answers.append(a_refine)

    if qtype == "mcq":
        # technique 5: option elimination
        answers.append(clean_answer(call_llm(prompt_eliminate(q))))
    elif qtype == "math":
        # technique 6: verification pass
        answers.append(clean_answer(call_llm(prompt_verify(q, draft), max_tokens=512)))

    # technique 7: self-consistency
    if disagree and qtype in {"math", "mcq"}:
        answers.append(clean_answer(call_llm(prompt_cot(q, qtype), temperature=0.4, max_tokens=tokens)))

    final = majority_vote(answers, qtype)

    if qtype == "mcq":
        return extract_mcq_letter(final) or final
    if qtype == "yesno":
        return normalize_yes_no(final)
    return final or ""