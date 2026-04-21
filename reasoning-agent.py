import os, json, re, time, subprocess, requests
from collections import Counter

api_key="sk-1F-lQI1P4uqRJ7lG72e2Gg"
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
    except requests.RequestException:
        return None