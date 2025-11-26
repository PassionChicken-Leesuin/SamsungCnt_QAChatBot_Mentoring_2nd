import os
from datetime import datetime, timedelta  # 🔹 timedelta 추가

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt   # 🔹 추가
from utils.faiss_downloader import ensure_faiss_index  # 🔹 인덱스 다운로드

# 🔹 RAG / 통계 유틸
from utils.rag_utils import (
    load_vectorstore,
    build_accident_stats,
    answer_with_stats_using_index,
    create_rag_chain,
)

# 🔹 뉴스 검색 유틸
from services.news_aggregator import collect_news
from utils.summarizer import summarize_text

# 🔹 말풍선 렌더링용
import html
import markdown as md


# ======================================================
# 0. 환경 설정 / 페이지 설정
# ======================================================
load_dotenv(override=True)

st.set_page_config(
    page_title="✨ Safety AI Mate : 안전부터 일상까지 함께해요",
    layout="wide",
)

model_name = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
embedding_model = "text-embedding-3-small"


# ======================================================
# 0-0. Q 오른쪽 / A 왼쪽 말풍선 HTML
# ======================================================
def build_bubble_html(role: str, content: str) -> str:
    """
    - user: 오른쪽, 남색 말풍선 (텍스트 escape)
    - assistant: 왼쪽, 연한 회색/화이트 말풍선 (Markdown 렌더링)
    """
    role = (role or "").lower()

    # assistant는 Markdown 렌더링, user는 plain text + 줄바꿈만
    if role in ["assistant", "ai", "bot"]:
        inner_html = md.markdown(
            content or "",
            extensions=["tables", "fenced_code"],
        )
    else:
        inner_html = html.escape(content or "").replace("\n", "<br>")

    if role in ["user", "human"]:
        # 오른쪽 정렬 (질문)
        return f"""
        <div class="chat-row chat-row-user">
          <div class="chat-bubble-wrapper chat-bubble-wrapper-user">
            <div class="chat-bubble chat-bubble-user">
              {inner_html}
            </div>
            <span class="chat-avatar chat-avatar-user">🆀</span>
          </div>
        </div>
        """
    else:
        # 왼쪽 정렬 (답변)
        return f"""
        <div class="chat-row chat-row-assistant">
          <div class="chat-bubble-wrapper chat-bubble-wrapper-assistant">
            <span class="chat-avatar chat-avatar-assistant">🅰</span>
            <div class="chat-bubble chat-bubble-assistant">
              {inner_html}
            </div>
          </div>
        </div>
        """


# ======================================================
# 0-1. 공통 CSS (삼성 블루톤 스타일 + 아이콘 깨짐 방지)
# ======================================================
st.markdown(
    """
    <style>
    /* 전체 배경 톤 약간 밝게 */
    .stApp {
        background-color: #F4F7FB;
    }

    /* 메인 컨텐츠 폭 약간 좁게 + 가운데 정렬 느낌 */
    .main .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* 사이드바 배경 단색(연한 회색) */
    [data-testid="stSidebar"] {
        background-color: #E9ECEF;
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 사이드바 상단 카드 공통 폰트 */
    .sidebar-title-card * {
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 🔵 사이드바 안의 버튼 전용 스타일 */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1248A8 !important;  /* 삼성 블루 */
        color: #FFFFFF !important;
        border: 1px solid #0F2F6A !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.8rem !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #0F2F6A !important;  /* hover 시 조금 더 진한 블루 */
        border-color: #091A3F !important;
    }

    /* 기본 버튼 스타일(메인 영역 등) */
    .stButton>button {
        background: #1428A0;
        color: white;
        border-radius: 999px;
        border: 1px solid #0F2F6A;
        padding: 0.4rem 0.8rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: #0F2F6A;
        border-color: #0A1E4F;
    }

    /* Expander(뉴스 검색 조건) 헤더 색상 톤 정리 */
    .streamlit-expanderHeader {
        font-family: 'Noto Sans KR', sans-serif;
        font-size: 16px;
        font-weight: 600;
        color: #1428A0 !important;
    }

    /* 🔵 사이드바 뉴스 소스 선택 멀티셀렉트 태그(선택된 값) 파스텔 블루 톤 */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #E8F1FF !important;  /* 파스텔 블루 배경 */
        color: #1248A8 !important;             /* 삼성 블루 텍스트 */
        border-radius: 999px !important;       /* 동글동글 pill 모양 */
        border: 1px solid #D4E3FF !important;  /* 연한 블루 테두리 */
        font-weight: 500;
    }

    /* 채팅 영역 컨테이너 */
    .chat-container {
        background-color: transparent;   /* 🔹 흰 배경 제거 */
        border-radius: 0;               /* 🔹 둥근 모서리 제거 */
        padding: 0;                     /* 🔹 안쪽 여백 제거 */
        box-shadow: none;               /* 🔹 그림자 제거 */
        margin-top: 0;                  /* 🔹 위/아래 여백 최소화 */
        margin-bottom: 0;
        max-height: none;               /* 🔹 스크롤 박스 느낌 제거 */
        overflow: visible;
    }

    .chat-row {
        display: flex;
        margin: 0.25rem 0;
    }

    .chat-bubble-wrapper {
        display: flex;
        align-items: flex-end;
        gap: 0.35rem;
        max-width: 80%;
    }

    .chat-row-user {
        justify-content: flex-end;
    }
    .chat-row-assistant {
        justify-content: flex-start;
    }

    .chat-avatar {
        font-size: 1.6rem;
        line-height: 1;
    }

    .chat-bubble {
        font-family: 'Noto Sans KR', sans-serif;
        font-size: 0.94rem;
        line-height: 1.5;
        word-wrap: break-word;
        word-break: break-word;
        padding: 0.6rem 0.9rem;
        border-radius: 1rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.12);
    }

    .chat-bubble-user {
        background-color: #1248A8;
        color: #FFFFFF;
        border-bottom-right-radius: 0.25rem;
        text-align: left;
    }

    .chat-bubble-assistant {
        background-color: #F9FAFB;
        color: #111827;
        border: 1px solid #E5E7EB;
        border-bottom-left-radius: 0.25rem;
        text-align: left;
    }

    /* assistant 말풍선 안의 코드블럭/표 스타일 조금 정리 */
    .chat-bubble-assistant pre {
        background-color: #111827;
        color: #F9FAFB;
        padding: 0.6rem 0.8rem;
        border-radius: 0.6rem;
        overflow-x: auto;
        font-size: 0.8rem;
    }
    .chat-bubble-assistant code {
        font-size: 0.84rem;
    }
    .chat-bubble-assistant table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.86rem;
    }
    .chat-bubble-assistant table, 
    .chat-bubble-assistant th, 
    .chat-bubble-assistant td {
        border: 1px solid #E5E7EB;
        padding: 0.35rem 0.45rem;
    }

    /* 🔧 Material Icons 명시적 import + 폰트 지정 */
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    .material-icons,
    [class^="material-icons"],
    [class*=" material-icons"] {
        font-family: 'Material Icons' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# 0-2. 사이드바 상단 카드
# ======================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title-card" style="
            background: linear-gradient(135deg, #E8F1FF 0%, #F7FAFF 100%;
            padding: 18px 14px;
            border-radius: 16px;
            border: 1px solid #D4E3FF;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            text-align: center;
            margin-bottom: 20px;
        ">
            <div style="
                font-size: 30px;
                font-weight: 800;
                color: #1248A8;
                margin-bottom: 6px;
            ">
                ✨ Safety AI Mate ✨
            </div>
            <div style="
                font-size: 18px;
                font-weight: 500;
                color: #44618A;
                margin-bottom: 6px;
            ">
                안전부터 일상까지 함께해요
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================
# 1. 세션 초기화 (모드별로 별도 저장)
# ======================================================
if "normal_messages" not in st.session_state:
    st.session_state["normal_messages"] = []

if "safe_messages" not in st.session_state:
    st.session_state["safe_messages"] = []

if "news_messages" not in st.session_state:
    st.session_state["news_messages"] = []

# 뉴스 자동 로딩 여부 플래그
if "news_first_load" not in st.session_state:
    st.session_state["news_first_load"] = True

# 오늘 뉴스 표 저장용
if "news_auto_df" not in st.session_state:
    st.session_state["news_auto_df"] = None


# ======================================================
# 2. 메뉴 / 현재 챗봇 선택
# ======================================================
st.sidebar.markdown(
    """
    <div style="
        font-family: 'Noto Sans KR', sans-serif;
        font-size: 22px;
        font-weight: 600;
        color: #1248A8;
        margin: 8px 0 4px 0;
    ">
        모드 선택
    </div>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "모드 선택",
    ["안전뉴스 검색/요약 Mate", "일상정보 Mate", "안전사고 검색 Mate(RAG)"],
    index=0,
)

# 현재 모드에 맞는 대화 키 선택
if menu == "일상정보 Mate":
    current_chat = "normal_messages"
elif menu == "안전사고 검색 Mate(RAG)":
    current_chat = "safe_messages"
else:  # "안전뉴스 검색/요약 Mate"
    current_chat = "news_messages"

# 현재 모드만 초기화
if st.sidebar.button("대화 초기화", use_container_width=True):
    st.session_state[current_chat] = []


# ======================================================
# 2-1. 메인 타이틀(모드별 설명)
# ======================================================
if menu == "안전뉴스 검색/요약 Mate":
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 30px;
                font-weight: 700;
                color: #1248A8;
                margin-bottom: 4px;
            ">
                📰 안전 인사이트, 1분 브리핑 : "안전뉴스 검색/요약 Mate"
            </div>
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 16px;
                font-weight: 400;
                color: #44618A;
            ">
                수많은 정보 속, 꼭 알아야 할 안전 이슈만 쏙쏙 뽑아 요약해 드립니다. 
                똑똑하고 든든한 당신의 안전 지킴이 'Mate'에게 무엇이든 물어보세요!
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif menu == "안전사고 검색 Mate(RAG)":
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 30px;
                font-weight: 700;
                color: #1248A8;
                margin-bottom: 4px;
            ">
                🚨 안전사고 백과사전 : "안전사고 검색 Mate"
            </div>
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 15px;
                font-weight: 400;
                color: #44618A;
            ">
                사전에 임베딩해둔 실제 안전사고 사례 데이터를 기반으로, 
                정확한 검색과 통계 기반 답변을 제공합니다. 
                부정확한 정보는 이제 그만! 가장 믿을 수 있는 안전 정보 Mate와 함께하세요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:  # 일상정보 Mate
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 30px;
                font-weight: 700;
                color: #1248A8;
                margin-bottom: 4px;
            ">
                💬 일상 속 모든 질문, 지금 바로 'Mate'에게
            </div>
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 15px;
                font-weight: 400;
                color: #44618A;
            ">
                생활 팁부터 최신 트렌드까지, 일상에 필요한 모든 정보를 친절하게 설명해 드립니다. 
                바쁜 하루를 돕는 가장 스마트한 비서 'Mate'입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================
# 공통 함수
# ======================================================
def add_message(role: str, content: str) -> None:
    """현재 모드의 대화에 메시지 추가"""
    st.session_state[current_chat].append(ChatMessage(role=role, content=content))


def print_messages() -> None:
    """현재 모드의 대화 전체 출력 (Q 오른쪽 / A 왼쪽 말풍선)"""
    msgs = st.session_state[current_chat]
    if not msgs:
        return  # 🔹 메시지 없으면 아무것도 렌더링 안 함

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in msgs:
        st.markdown(
            build_bubble_html(msg.role, msg.content),
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def get_history_text(max_turns: int = 5) -> str:
    """
    RAG용 히스토리.
    - 안전사고 검색 Mate(RAG)에서만 사용하므로 safe_messages 기준으로 작성
    """
    msgs = st.session_state.get("safe_messages", [])
    turns = [m for m in msgs if m.role in ("user", "assistant")]
    turns = turns[-max_turns * 2 :]

    history = []
    for msg in turns:
        role = "사용자" if msg.role == "user" else "AI"
        history.append(f"{role}: {msg.content}")
    return "\n".join(history)


# ======================================================
# 3. 일반 GPT 체인
# ======================================================
def create_normal_chain():
    def build_prompt(inputs):
        return f"""
        당신은 매우 친절한 AI 어시스턴트입니다.

        질문:
        {inputs["question"]}
        """

    llm = ChatOpenAI(model=model_name, temperature=0)

    return RunnableLambda(build_prompt) | llm | StrOutputParser()


# ======================================================
# 3-1. 뉴스 요약/답변 체인
# ======================================================
def create_news_chain():
    prompt = load_prompt("prompts/news.yaml", encoding="utf-8")
    llm = ChatOpenAI(model=model_name, temperature=0)
    return prompt | llm | StrOutputParser()


# ======================================================
# 4. FAISS + 통계 로딩
# ======================================================
@st.cache_resource
def get_vectorstore_and_stats():
    # 🔹 1) 구글드라이브에서 인덱스 파일 다운(or 재사용)
    ensure_faiss_index(index_dir="faiss_index")

    # 🔹 2) FAISS 인덱스 로딩
    store, docs = load_vectorstore(
        embedding_model,
        index_path="faiss_index",
    )

    # 🔹 3) 통계 계산
    stats = build_accident_stats(docs)
    return store, stats, docs


# ======================================================
# 5. 뉴스 챗봇 — 사이드바 옵션 (+ 기간 설정 UI)
# ======================================================
news_sources = None
news_max_items = None
news_start_date = None
news_end_date = None
run_news_search = False

if menu == "안전뉴스 검색/요약 Mate":

    with st.sidebar.expander("뉴스 검색 조건", expanded=True):
        # 뉴스 소스 선택
        st.markdown(
            """
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 16px;
                font-weight: 600;
                color: #44618A;
                margin: 4px 0 4px 0;
            ">
                ⚪ 뉴스 소스 선택
            </div>
            """,
            unsafe_allow_html=True,
        )
        news_sources = st.multiselect(
            "뉴스 소스 선택",
            ["Google", "Naver", "Kakao", "Tavily", "SerpAPI"],
            default=["Google", "Naver"],
            label_visibility="collapsed",
        )

        # 소스별 최대 기사 수
        st.markdown(
            """
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 16px;
                font-weight: 600;
                color: #44618A;
                margin: 12px 0 4px 0;
            ">
                ⚪ 소스별 최대 기사 수
            </div>
            """,
            unsafe_allow_html=True,
        )
        news_max_items = st.slider(
            "소스별 최대 기사 수",
            1,
            10,
            3,
            label_visibility="collapsed",
        )

        # 검색 기간
        st.markdown(
            """
            <div style="
                font-family: 'Noto Sans KR', sans-serif;
                font-size: 16px;
                font-weight: 600;
                color: #44618A;
                margin: 12px 0 4px 0;
            ">
                ⚪ 검색 기간
            </div>
            """,
            unsafe_allow_html=True,
        )

        date_mode = st.radio(
            "기간 선택 방식",
            ["최근 N일", "사용자 지정"],
            index=0,
            label_visibility="collapsed",
        )

        today = datetime.today().date()

        if date_mode == "최근 N일":
            days_range = st.slider(
                "오늘부터 며칠 전까지 볼까요?",
                1,
                7,
                2,
                label_visibility="collapsed",
            )
            news_start_date = today - timedelta(days=days_range)
            news_end_date = today
            st.caption(f"📌 선택된 기간: {news_start_date} ~ {news_end_date}")
        else:
            col1, col2 = st.columns(2)
            news_start_date = col1.date_input("시작일", today - timedelta(days=7))
            news_end_date = col2.date_input("종료일", today)
            st.caption(f"📌 선택된 기간: {news_start_date} ~ {news_end_date}")

    run_news_search = st.sidebar.button(
        "🔍 검색 시작",
        use_container_width=True,
    )

else:
    news_sources = None
    news_max_items = None
    news_start_date = None
    news_end_date = None
    run_news_search = False


# ======================================================
# 6. 앱 진입 시 1회 자동 뉴스 요약
# ======================================================
if menu == "안전뉴스 검색/요약 Mate" and st.session_state["news_first_load"]:
    query = "안전사고"

    with st.spinner("⏳ 최신 뉴스 자동 수집 중..."):
        try:
            auto_df = collect_news(
                query=query,
                sources=["Google", "Naver", "Kakao", "Tavily", "SerpAPI"],
                max_items=3,  # 🔹 각 소스별 최대 3개만
            )
            if not auto_df.empty:
                auto_df = auto_df.dropna(axis=1, how="all")
        except Exception as e:
            st.warning(f"뉴스 수집 중 오류: {e}")
            auto_df = pd.DataFrame()

    if auto_df.empty:
        st.warning("⚠ 수집된 뉴스가 없습니다.")
        st.session_state["news_auto_df"] = None
    else:
        today = datetime.now().date()

        # 🔹 collect_news에서 최대한 'published'로 맞춰왔다는 전제
        if "published" in auto_df.columns:
            # 1) published를 한 번 더 안전하게 datetime으로 변환
            if not pd.api.types.is_datetime64_any_dtype(auto_df["published"]):
                auto_df["published"] = pd.to_datetime(
                    auto_df["published"],
                    errors="coerce",
                )

            # 2) 여전히 datetime 타입이면 .dt 사용, 아니면 그냥 전체 사용
            if pd.api.types.is_datetime64_any_dtype(auto_df["published"]):
                # ✅ 오늘 날짜 + 날짜 없는(NaT) 기사도 포함
                mask_today = auto_df["published"].dt.date == today
                mask_na = auto_df["published"].isna()
                today_df = auto_df[mask_today | mask_na].copy()

                # 그래도 비어 있으면(오늘/NaT 모두 없으면) 최근 10개로 대체
                if today_df.empty:
                    today_df = (
                        auto_df.sort_values("published", ascending=False)
                        .head(10)
                        .copy()
                    )
            else:
                # datetime으로 못 바꾸면 그냥 상위 10개
                today_df = auto_df.head(10).copy()
        else:
            # 날짜 컬럼이 아예 없으면 그냥 상위 10개
            today_df = auto_df.head(10).copy()

        if today_df.empty:
            st.warning("⚠ 오늘 날짜에 해당하는 뉴스가 없습니다.")
            st.session_state["news_auto_df"] = None
        else:
            base_cols = ["source", "title", "url", "published"]
            show_cols = [c for c in base_cols if c in today_df.columns]
            if show_cols:
                display_df = today_df[show_cols]
            else:
                display_df = today_df

            # 👉 사이드바 표에 보여줄 데이터 저장
            st.session_state["news_auto_df"] = display_df

            possible_cols = ["title"]
            text_col = next((c for c in possible_cols if c in today_df.columns), None)

            if text_col:

                def build_summary_input(row):
                    title = str(row.get("title", ""))
                    return f"{title}"

                today_df["summary_input"] = today_df.apply(
                    build_summary_input, axis=1
                )
                today_df["summary"] = today_df["summary_input"].astype(str).apply(
                    summarize_text
                )

                lines = []
                for _, row in today_df.iterrows():
                    src = row.get("source", "")
                    title = row.get("title", "")
                    summ = row.get("summary", "")
                    line = f"- [{src}] {title}\n  요약: {summ}"
                    lines.append(line)

                initial_answer = (
                    "📡 최신 안전사고 뉴스 자동 요약입니다:\n\n"
                    + "\n\n".join(lines)
                )
                add_message("assistant", initial_answer)
            else:
                st.error("요약할 텍스트 컬럼을 찾지 못했습니다.")
                st.session_state["news_auto_df"] = None

    st.session_state["news_first_load"] = False


# ======================================================
# 6-1. 오늘자 자동 뉴스 블록 렌더링
# ======================================================
def render_auto_news_block():
    if menu != "안전뉴스 검색/요약 Mate":
        return

    auto_df = st.session_state.get("news_auto_df")
    if auto_df is None or auto_df.empty:
        return

    st.subheader("📡 오늘의 안전 사고 뉴스")
    st.dataframe(auto_df, hide_index=True)
    st.subheader("📝 오늘의 뉴스 요약")


render_auto_news_block()


# ======================================================
# 7. 기존 대화 출력 (현재 모드만)
# ======================================================
print_messages()


# ======================================================
# 7-1. 🔍 '검색 시작' 버튼 기반 즉시 뉴스 검색/요약
# ======================================================
if menu == "안전뉴스 검색/요약 Mate" and run_news_search:

    fake_user_query = "기본 질의 '안전사고'로 최신 뉴스를 검색합니다."
    # 질문 말풍선
    st.markdown(
        '<div class="chat-container">'
        + build_bubble_html("user", fake_user_query)
        + "</div>",
        unsafe_allow_html=True,
    )
    add_message("user", fake_user_query)

    container = st.empty()
    ai_answer = ""

    query = "안전사고"
    sources = news_sources or ["Google", "Naver"]
    max_items = news_max_items or 10

    with st.spinner("⏳ 뉴스 검색 중..."):
        try:
            df = collect_news(
                query=query,
                sources=sources,
                max_items=max_items,
                start_date=news_start_date,
                end_date=news_end_date,
            )
        except Exception as e:
            df = pd.DataFrame()
            container.markdown(
                '<div class="chat-container">'
                + build_bubble_html("assistant", f"뉴스 검색 중 오류: {e}")
                + "</div>",
                unsafe_allow_html=True,
            )

    if df.empty:
        answer = "해당 조건에 맞는 뉴스가 없습니다."
        for ch in answer:
            ai_answer += ch
            container.markdown(
                '<div class="chat-container">'
                + build_bubble_html("assistant", ai_answer)
                + "</div>",
                unsafe_allow_html=True,
            )
    else:
        if "published" in df.columns:
            df["published"] = pd.to_datetime(df["published"], errors="coerce")

        show_cols = [
            c
            for c in ["source", "title", "url", "published"]
            if c in df.columns
        ]
        if show_cols:
            st.subheader("📊 검색된 뉴스 목록")
            st.dataframe(df[show_cols], hide_index=True)

        def build_summary_input(row):
            title = str(row.get("title", ""))
            url = str(row.get("url", ""))
            return f"{title}\n\n기사 링크: {url}"

        df["summary_input"] = df.apply(build_summary_input, axis=1)
        df["summary"] = df["summary_input"].astype(str).apply(summarize_text)

        article_lines = []
        for _, row in df.head(max_items).iterrows():
            src = row.get("source", "")
            title = row.get("title", "")
            url = row.get("url", "")
            published = row.get("published", "")
            summ = row.get("summary", "")

            line = (
                f"[출처] {src}\n"
                f"[제목] {title}\n"
                f"[발행일] {published}\n"
                f"[링크] {url}\n"
                f"[요약] {summ}"
            )
            article_lines.append(line)

        articles_text = "\n\n-----\n\n".join(article_lines)

        news_chain = create_news_chain()
        for token in news_chain.stream(
            {
                "question": query,
                "context": articles_text,
            }
        ):
            ai_answer += token
            container.markdown(
                '<div class="chat-container">'
                + build_bubble_html("assistant", ai_answer)
                + "</div>",
                unsafe_allow_html=True,
            )

    add_message("assistant", ai_answer)


# ======================================================
# 8. 채팅 입력 (세 모드 공통, stream 출력)
# ======================================================
user_input = st.chat_input("질문을 입력하세요")

if user_input:
    # 1) 사용자 말풍선 출력 + 저장
    st.markdown(
        '<div class="chat-container">'
        + build_bubble_html("user", user_input)
        + "</div>",
        unsafe_allow_html=True,
    )
    add_message("user", user_input)

    # 2) 어시스턴트 답변 말풍선 (stream)
    container = st.empty()
    ai_answer = ""

    # ---------------------------
    # 일상정보 Mate (일반 GPT 모드)
    # ---------------------------
    if menu == "일상정보 Mate":
        chain = create_normal_chain()
        for token in chain.stream({"question": user_input}):
            ai_answer += token
            container.markdown(
                '<div class="chat-container">'
                + build_bubble_html("assistant", ai_answer)
                + "</div>",
                unsafe_allow_html=True,
            )

    # ---------------------------
    # 안전사고 검색 Mate(RAG) 모드
    # ---------------------------
    elif menu == "안전사고 검색 Mate(RAG)":
        vectorstore, accident_stats, index_docs = get_vectorstore_and_stats()

        stats_answer = answer_with_stats_using_index(
            user_input, accident_stats, index_docs
        )

        if stats_answer is not None:
            for ch in stats_answer:
                ai_answer += ch
                container.markdown(
                    '<div class="chat-container">'
                    + build_bubble_html("assistant", ai_answer)
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            rag_chain = create_rag_chain(
                vectorstore=vectorstore,
                rag_prompt_path="prompts/first.yaml",
                llm_model_name=model_name,
            )
            for token in rag_chain.stream(
                {
                    "question": user_input,
                    "history": get_history_text(),
                }
            ):
                ai_answer += token
                container.markdown(
                    '<div class="chat-container">'
                    + build_bubble_html("assistant", ai_answer)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # ---------------------------
    # 안전뉴스 검색/요약 Mate
    # ---------------------------
    else:
        query = user_input
        sources = news_sources or ["Google", "Naver"]
        max_items = news_max_items or 10

        with st.spinner("⏳ 뉴스 검색 중..."):
            try:
                df = collect_news(
                    query=query,
                    sources=sources,
                    max_items=max_items,
                    start_date=news_start_date,
                    end_date=news_end_date,
                )

            except Exception as e:
                df = pd.DataFrame()
                st.warning(f"뉴스 검색 중 오류: {e}")

        if df.empty:
            answer = "관련 뉴스가 없습니다."
            for ch in answer:
                ai_answer += ch
                container.markdown(
                    '<div class="chat-container">'
                    + build_bubble_html("assistant", ai_answer)
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            if "published" in df.columns:
                df["published"] = pd.to_datetime(df["published"], errors="coerce")

            df_display = df.copy()

            if "published" not in df_display.columns:
                df_display["published"] = pd.NaT

            show_cols = [
                c
                for c in ["source", "title", "url", "published"]
                if c in df_display.columns
            ]

            if show_cols:
                st.subheader("📊 수집된 뉴스 목록")
                st.dataframe(df_display[show_cols], hide_index=True)

            def build_summary_input(row):
                title = str(row.get("title", ""))
                url = str(row.get("url", ""))
                return f"{title}\n\n기사 링크: {url}"

            df["summary_input"] = df.apply(build_summary_input, axis=1)
            df["summary"] = df["summary_input"].astype(str).apply(summarize_text)

            article_lines = []
            for _, row in df.head(max_items).iterrows():
                src = row.get("source", "")
                title = row.get("title", "")
                url = row.get("url", "")
                published = row.get("published", "")
                summ = row.get("summary", "")

                line = (
                    f"[출처] {src}\n"
                    f"[제목] {title}\n"
                    f"[발행일] {published}\n"
                    f"[링크] {url}\n"
                    f"[요약] {summ}"
                )
                article_lines.append(line)

            articles_text = "\n\n-----\n\n".join(article_lines)

            news_chain = create_news_chain()
            for token in news_chain.stream(
                {
                    "question": user_input,
                    "context": articles_text,
                }
            ):
                ai_answer += token
                container.markdown(
                    '<div class="chat-container">'
                    + build_bubble_html("assistant", ai_answer)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # 3) 대화 기록 저장
    add_message("assistant", ai_answer)
