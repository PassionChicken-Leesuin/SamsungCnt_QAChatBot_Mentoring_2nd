import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# .env 로드
load_dotenv()

# .env 에 SERPAPI_API_KEY=... 형태로 들어있다고 가정
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")


def fetch_serpapi_news(
    query: str = "산업 안전 사고",
    num: int = 10,
    debug: bool = False
) -> dict:
    """
    SerpAPI 기반 Google News 검색

    Parameters
    ----------
    query : str
        검색어 (예: '산업 안전 사고')
    num : int
        가져올 뉴스 개수 (SerpAPI의 num 파라미터 그대로 사용)
    debug : bool
        True일 경우 상태코드 / 요청 URL / 응답 일부 출력

    Returns
    -------
    dict
        SerpAPI JSON 응답
    """

    if SERPAPI_KEY is None:
        raise ValueError("❗ SERPAPI_API_KEY가 .env에 설정되어 있지 않습니다!")

    url = "https://serpapi.com/search"

    params = {
        "engine": "google_news",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num,      # 🔥 여기 값이 max_items 그대로 반영됨
        "gl": "kr",
        "hl": "ko",
    }

    resp = requests.get(url, params=params)

    if debug:
        print("🔍 status:", resp.status_code)
        print("🔍 요청 URL:", resp.url)
        print("\n----- 응답 원문 (앞 500자) -----")
        print(resp.text[:500])

    resp.raise_for_status()
    return resp.json()


# 🔥 SerpAPI 날짜 포맷 전용 파서
def parse_serpapi_date(date_str):
    """
    예시: '11/25/2025, 01:47 AM, +0000 UTC'
    → datetime 객체로 변환
    """
    if not isinstance(date_str, str):
        return None

    # ' UTC' 제거 → '11/25/2025, 01:47 AM, +0000'
    cleaned = date_str.replace(" UTC", "")

    # 포맷: MM/DD/YYYY, HH:MM AM/PM, +0000
    try:
        return datetime.strptime(cleaned, "%m/%d/%Y, %I:%M %p, %z")
    except Exception:
        return None


def serpapi_to_df(data: dict) -> pd.DataFrame:
    """
    SerpAPI Google News JSON → DataFrame 변환

    Parameters
    ----------
    data : dict
        fetch_serpapi_news 응답 JSON

    Returns
    -------
    DataFrame
        title / url / published / content / source 가 담긴 DataFrame
        (없는 컬럼은 자동으로 제외)
    """

    items = data.get("news_results", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    # source 컬럼이 dict 형태일 수 있으므로 이름만 추출
    if "source" in df.columns:
        df["source"] = df["source"].apply(
            lambda s: s.get("name") if isinstance(s, dict) else s
        )

    # 공통 컬럼명으로 통일
    rename_map = {}
    if "link" in df.columns:
        rename_map["link"] = "url"
    if "date" in df.columns and "published" not in df.columns:
        rename_map["date"] = "published"
    if "snippet" in df.columns and "content" not in df.columns:
        rename_map["snippet"] = "content"

    if rename_map:
        df = df.rename(columns=rename_map)

    # 🔍 디버그: 원본 published 문자열 확인
    if "published" in df.columns:
        print("[SERPAPI DEBUG] published raw head ===")
        print(df["published"].head())
        print("[SERPAPI DEBUG] published raw dtype:", df["published"].dtype)

        # 🔥 여기서 직접 파싱 (절대 pd.to_datetime() 다시 쓰지 않기!)
        df["published"] = df["published"].apply(parse_serpapi_date)

        print("\n[SERPAPI DEBUG] published parsed head ===")
        print(df["published"].head())
        print("[SERPAPI DEBUG] published parsed dtype:", df["published"].dtype)

    # 최종적으로 자주 쓰는 컬럼만 정리해서 반환
    cols = ["title", "url", "published", "content", "source"]
    cols = [c for c in cols if c in df.columns]

    return df[cols]
