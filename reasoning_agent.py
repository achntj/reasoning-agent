import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = "https://openai.rc.asu.edu/v1"
model = "qwen3-30b-a3b-instruct-2507"


def call_llm(
    prompt, system="You are a helpful assistant.", temperature=0.0, max_tokens=256
):
    if not api_key:
        raise ValueError("Missing API_KEY in environment.")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        print("API error:", response.status_code, response.text)
        return ""
    except Exception as e:
        print("API request failed:", e)
        return ""


def clean_answer(text):
    if not text:
        return ""

    text = text.strip()

    patterns = [
        r"(?i)final answer\s*[:\-]\s*(.*)",
        r"(?i)answer\s*[:\-]\s*(.*)",
        r"(?i)output\s*[:\-]\s*(.*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()
            break

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    return text[:4999].strip()


def build_direct_prompt(question_text):
    return f"""Answer the following question.

Return only the final answer as a plain string.
Do not include reasoning or explanation.

Question: {question_text}
"""


def build_cot_prompt(question_text):
    return f"""Solve the following question carefully step by step internally.

Return only the final answer.
Do not include your reasoning.

Question: {question_text}
"""


def build_decomposition_prompt(question_text):
    return f"""Break this question into smaller parts internally, solve it carefully, and combine the result.

Return only the final answer.
Do not include any explanation.

Question: {question_text}
"""


def build_self_refine_prompt(question_text, draft_answer):
    return f"""You previously answered this question.

Question: {question_text}

Previous answer: {draft_answer}

Check the answer and correct it if needed.

Return only the final corrected answer.
Do not include explanation.
"""


def answer_question(question_text):
    question_text = question_text.strip()

    # 1. Direct prompting
    direct_answer = clean_answer(
        call_llm(build_direct_prompt(question_text), temperature=0.0)
    )

    # 2. Hidden chain-of-thought prompting
    cot_answer = clean_answer(
        call_llm(build_cot_prompt(question_text), temperature=0.0)
    )

    # 3. Decomposition prompting
    decomposition_answer = clean_answer(
        call_llm(build_decomposition_prompt(question_text), temperature=0.0)
    )

    # pick a simple draft answer
    draft_answer = decomposition_answer or cot_answer or direct_answer

    # 4. Self-refine
    final_answer = clean_answer(
        call_llm(build_self_refine_prompt(question_text, draft_answer), temperature=0.0)
    )

    if final_answer:
        return final_answer
    if draft_answer:
        return draft_answer
    if cot_answer:
        return cot_answer
    return direct_answer
