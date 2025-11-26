# services/news_aggregator.py

import pandas as pd

import fetchers.google_rss as google_rss
import fetchers.naver_news as naver_news
import fetchers.kakao_news as kakao_news
import fetchers.tavily as tavily
import fetchers.serpapi as serpapi


def _normalize_common_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    각 소스별로 제각각인 컬럼명을 최대한 통일하고,
    불필요한 link 컬럼 등을 정리하는 헬퍼 함수.
    - URL: url
    - 날짜: published
    - 요약(있으면): summary

    ⚠ 일부 fetcher(예: SerpAPI)는 이미 'published'를 datetime으로 만들어서 주므로,
      여기서는 'published'가 이미 있으면 그대로 두고 추가 rename/drop 정도만 수행.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["source"] = source

    # ---------- URL 컬럼 통일 ----------
    # 우선순위: url > link > linkUrl ...
    if "url" not in df.columns:
        if "link" in df.columns:
            df = df.rename(columns={"link": "url"})
        elif "linkUrl" in df.columns:
            df = df.rename(columns={"linkUrl": "url"})

    # url이 생겼는데 link가 남아 있으면 삭제
    if "url" in df.columns and "link" in df.columns:
        df = df.drop(columns=["link"])

    # ---------- 날짜 컬럼 통일 ----------
    # 이미 fetcher에서 'published'를 만들어 줬으면 그대로 두고,
    # pubDate / datetime / date 같은 raw 컬럼은 정리만 해준다.
    if "published" in df.columns:
        # 원시 날짜 컬럼은 있으면 정리 (선택 사항)
        for c in ["pubDate", "datetime", "date"]:
            if c in df.columns:
                df = df.drop(columns=[c])
    else:
        # published가 없으면 pubDate / datetime / date 중 하나를 published로 승격
        for c in ["pubDate", "datetime", "date"]:
            if c in df.columns:
                df = df.rename(columns={c: "published"})
                break

    # ---------- 요약 컬럼 통일 (있으면) ----------
    # Google RSS는 기존 로직대로 description -> summary
    if "description" in df.columns and "summary" not in df.columns:
        df = df.rename(columns={"description": "summary"})

    return df


def collect_news(
    query,
    sources,
    max_items: int = 10,
    start_date=None,   # 🔹 기간 필터 시작일 (date/datetime/str 모두 허용)
    end_date=None,     # 🔹 기간 필터 종료일
) -> pd.DataFrame:
    """
    여러 소스에서 뉴스를 수집하고 하나의 DataFrame으로 합친다.
    - 각 소스별로 최대 max_items개까지 수집
    - 공통 컬럼:
        * source   : 뉴스 출처 (Google / Naver / Kakao / Tavily / SerpAPI)
        * url      : 기사 URL
        * published: 발행일(가능한 경우)
        * summary  : (일부 소스에서 제공하는 요약/description)
    - start_date, end_date가 주어지면 published 기준으로 기간 필터링
    """
    dfs = []

    # ---------------- Google RSS ----------------
    if "Google" in sources:
        try:
            g_df = google_rss.fetch_google_news_rss(query, max_items=max_items)
            g_df = _normalize_common_columns(g_df, "Google")
            dfs.append(g_df)
        except Exception as e:
            print(f"[Google] 뉴스 수집 중 오류: {e}")

    # ---------------- Naver ----------------
    if "Naver" in sources:
        try:
            data = naver_news.search_naver_news(query, display=max_items)
            n_df = naver_news.naver_news_to_df(data)
            n_df = _normalize_common_columns(n_df, "Naver")
            dfs.append(n_df)
        except Exception as e:
            print(f"[Naver] 뉴스 수집 중 오류: {e}")

    # ---------------- Kakao ----------------
    if "Kakao" in sources:
        try:
            data = kakao_news.fetch_kakao_web(query, size=max_items)
            k_df = kakao_news.kakao_web_to_df(data)
            k_df = _normalize_common_columns(k_df, "Kakao")
            dfs.append(k_df)
        except Exception as e:
            print(f"[Kakao] 뉴스 수집 중 오류: {e}")

    # ---------------- Tavily ----------------
    if "Tavily" in sources:
        try:
            data = tavily.fetch_tavily_news(query, max_results=max_items)
            t_df = tavily.tavily_to_df(data)
            t_df = _normalize_common_columns(t_df, "Tavily")
            dfs.append(t_df)
        except Exception as e:
            print(f"[Tavily] 뉴스 수집 중 오류: {e}")

    # ---------------- SerpAPI ----------------
    if "SerpAPI" in sources:
        try:
            data = serpapi.fetch_serpapi_news(query, num=max_items)
            s_df = serpapi.serpapi_to_df(data)
            s_df = _normalize_common_columns(s_df, "SerpAPI")
            dfs.append(s_df)
        except Exception as e:
            print(f"[SerpAPI] 뉴스 수집 중 오류: {e}")

    # 아무 소스도 성공적으로 수집 못한 경우
    if not dfs:
        return pd.DataFrame()

    # 하나로 합치기
    all_df = pd.concat(dfs, ignore_index=True, sort=False)

    # 전체가 NaN인 컬럼은 제거 (잡스러운 컬럼 정리)
    all_df = all_df.dropna(axis=1, how="all")

    # ==================================================
    # 🔹 기간 필터링 (start_date / end_date가 주어졌을 때)
    #   - published를 날짜(date) 단위로만 비교 (시간/타임존은 무시)
    #   - 날짜가 없는(NaT) 기사들은 항상 포함
    # ==================================================
    if (start_date is not None or end_date is not None) and ("published" in all_df.columns):
        # 1) published를 datetime으로 일단 통일 시도
        if not pd.api.types.is_datetime64_any_dtype(all_df["published"]):
            all_df["published"] = pd.to_datetime(all_df["published"], errors="coerce")

        # 2) 여전히 datetime 타입이 아니면, 필터링 포기하고 그대로 반환
        if not pd.api.types.is_datetime64_any_dtype(all_df["published"]):
            return all_df

        pub = all_df["published"]
        # 🔹 날짜만 뽑기 (YYYY-MM-DD)
        pub_date = pub.dt.date
        has_date = pub.notna()
        no_date = pub.isna()

        # 3) 경계값도 date 타입으로 맞추기 (혹시 datetime/Timestamp로 들어오는 경우 대비)
        from datetime import datetime as _dt
        from pandas import Timestamp as _Ts

        sd = start_date
        ed = end_date

        if isinstance(sd, (_dt, _Ts)):
            sd = sd.date()
        if isinstance(ed, (_dt, _Ts)):
            ed = ed.date()

        # 4) 조건 만들기 (날짜 있는 애들만 범위 비교, 날짜 없는 애들은 무조건 포함)
        cond = pd.Series(True, index=all_df.index)

        if sd is not None:
            cond &= pub_date >= sd
        if ed is not None:
            cond &= pub_date <= ed

        mask = no_date | (has_date & cond)
        all_df = all_df[mask]

    return all_df

