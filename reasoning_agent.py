import os, json, re, time, subprocess, requests
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

#if we wanna keep api key a secret
#api_key= os.getenv("API_KEY") 

api_key= "sk-1F-lQI1P4uqRJ7lG72e2Gg"
base_url="https://openai.rc.asu.edu/v1"
model="qwen3-30b-a3b-instruct-2507"

def call_llm(prompt, system="You are a helpful assistant.", temperature=0.0):
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return None
    except Exception as e:
        print(f"API request failed: {e}")
        return None

#will edit to fit the 8 technique requirement later
def answer_question(question_text):
    prompt = f"""Answer the following question.
    Return only the final answer as a plain string.
    Do not include any intermediate reasoning steps or explanations, only the answer.
    Question: {question_text}
    """

    response = call_llm(prompt)

    if response is None:
        return ""
    return response.strip()