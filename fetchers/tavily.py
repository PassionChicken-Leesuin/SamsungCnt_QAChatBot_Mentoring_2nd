import os
import requests
import pandas as pd
from dotenv import load_dotenv

# .env 불러오기
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def fetch_tavily_news(query="산업 안전 사고", max_results=10, debug=False):
    """
    Tavily Search API (AI 기반 뉴스/웹 검색)
    """
    if TAVILY_API_KEY is None:
        raise ValueError("❗ TAVILY_API_KEY가 .env에 설정되어 있지 않습니다!")

    url = "https://api.tavily.com/search"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }

    payload = {
        "query": query,
        "max_results": max_results,
        "include_images": False,
        "include_answer": False,
        "search_depth": "advanced",
    }

    resp = requests.post(url, headers=headers, json=payload)
    print("🔍 status:", resp.status_code)
    print("🔍 요청 URL:", url)

    if debug:
        print("\n----- 응답 원문 (앞 500자) -----")
        print(resp.text[:500])

    resp.raise_for_status()
    return resp.json()


def tavily_to_df(data):
    """
    Tavily Search API JSON → DataFrame 변환
    + published (검색 시점 날짜) 컬럼 생성
    """
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    cols = ["title", "url", "content", "score"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df["published"] = pd.Timestamp.today().normalize()

    print("\n[TAVILY DEBUG] published head ===")
    print(df["published"].head())
    print("[TAVILY DEBUG] published dtype:", df["published"].dtype)

    return df

