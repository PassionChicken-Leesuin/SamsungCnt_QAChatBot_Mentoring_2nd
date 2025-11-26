import os
import requests
import pandas as pd
from dotenv import load_dotenv

# .env에서 KAKAO_REST_API_KEY 불러오기
load_dotenv()
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")


def fetch_kakao_web(query="산업 안전 사고", size=20, page=1, sort="recency"):
    """
    카카오 웹 문서 검색 API
    """
    if KAKAO_REST_API_KEY is None:
        raise ValueError("❗ KAKAO_REST_API_KEY가 .env에 설정되지 않았습니다!")

    url = "https://dapi.kakao.com/v2/search/web"

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}",
    }

    params = {
        "query": query,
        "size": size,
        "page": page,
        "sort": sort,
    }

    resp = requests.get(url, headers=headers, params=params)
    print("status:", resp.status_code)
    print("요청 URL:", resp.url)

    resp.raise_for_status()
    return resp.json()


def kakao_web_to_df(data):
    """
    카카오 웹 검색 JSON → DataFrame 변환
    + datetime → published (datetime)
    """
    docs = data.get("documents", [])
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    keep_cols = ["title", "contents", "url", "datetime"]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

    print("\n[KAKAO DEBUG] datetime head ===")
    print(df["datetime"].head())
    print("[KAKAO DEBUG] datetime dtype:", df["datetime"].dtype)

    df["published"] = pd.to_datetime(df["datetime"], errors="coerce")

    print("\n[KAKAO DEBUG] published head ===")
    print(df["published"].head())
    print("[KAKAO DEBUG] published dtype:", df["published"].dtype)

    return df

