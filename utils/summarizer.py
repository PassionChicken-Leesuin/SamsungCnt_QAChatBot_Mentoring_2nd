# utils/summarizer.py

import os
import requests
from dotenv import load_dotenv

load_dotenv(".env", override=True)

API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def summarize_text(text, max_sentences=3):
    """
    OpenAI Chat Completions API 기반 요약 함수
    """

    if not text or text.strip() == "":
        return "(요약 불가)"

    prompt = f"""
    다음 뉴스를 {max_sentences}문장으로 요약해줘.
    핵심적인 '안전 이슈'를 중심으로 간결하게 정리해줘.

    뉴스:
    {text}
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",   # 원하는 모델로 변경 가능
        "messages": [
            {"role": "system", "content": "넌 안전 사고/리스크 요약 전문 분석가야."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2
    }

    try:
        res = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()

        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"(요약 오류: {e})"
