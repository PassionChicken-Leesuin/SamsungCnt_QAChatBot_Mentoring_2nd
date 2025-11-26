import os
import requests
import pandas as pd
from dotenv import load_dotenv

# .env 불러오기
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")


def search_naver_news(query, display=20, start=1, sort="date", debug=False):
    """
    네이버 뉴스 검색 API 호출
    """
    if NAVER_CLIENT_ID is None or NAVER_CLIENT_SECRET is None:
        raise ValueError("❗ NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET이 .env에 없습니다!")

    url = "https://openapi.naver.com/v1/search/news.json"

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": sort,
    }

    resp = requests.get(url, headers=headers, params=params)

    print("🔍 status:", resp.status_code)
    print("🔍 요청 URL:", resp.url)

    if debug:
        print("\n----- 응답 원문 (앞 500자) -----")
        print(resp.text[:500])

    resp.raise_for_status()
    return resp.json()


def naver_news_to_df(data):
    """
    네이버 뉴스 검색 JSON → DataFrame 변환
    + pubDate를 datetime으로 파싱해서 published 컬럼 생성
    """
    items = data.get("items", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    # 원하는 주요 컬럼 골라내기
    keep_cols = ["title", "link", "pubDate", "description"]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

    # 🔹 pubDate 원본 확인
    print("\n[NAVER DEBUG] pubDate head ===")
    print(df["pubDate"].head())
    print("[NAVER DEBUG] pubDate dtype:", df["pubDate"].dtype)

    # 🔹 pubDate → published (datetime)
    df["published"] = pd.to_datetime(df["pubDate"], errors="coerce")

    print("\n[NAVER DEBUG] published head ===")
    print(df["published"].head())
    print("[NAVER DEBUG] published dtype:", df["published"].dtype)

    return df
