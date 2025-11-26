# utils/rag_utils.py

from typing import Dict, List, Tuple
from copy import deepcopy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import load_prompt, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap


# -----------------------------
# 1) 공통 유틸 / 상수들
# -----------------------------
def normalize_str(x):
    return str(x or "").strip()


REGION_CANDIDATES = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]

FORKLIFT_KEYWORDS = ["지게차", "포크리프트", "forklift"]

EQUIPMENT_CATEGORIES = {
    "지게차": ["지게차", "포크리프트", "forklift"],
    "크레인": ["크레인", "천장크레인", "천정크레인", "이동식크레인", "이동식 크레인"],
    "타워크레인": ["타워크레인", "타워 크레인"],
    "굴삭기": ["굴삭기", "포크레인", "포클레인", "백호", "excavator", "backhoe"],
    "변압기": ["변압기", "변전실", "트랜스", "변전소"],
    "트럭/차량": ["트럭", "화물차", "승용차", "차량", "덤프트럭", "지게차운행", "차량운행", "후진중"],
    "컨베이어": ["컨베이어", "conveyor"],
    "프레스": ["프레스", "press"],
    "기계톱": ["기계톱", "전기톱", "체인톱", "전동톱"],
    "사다리": ["사다리"],
    "리프트": ["리프트", "고소작업대"],
    "호이스트": ["호이스트"],
    "로프/와이어": ["와이어로프", "와이어 로프", "로프"],
    "압축기": ["압축기", "compressor"],
}

ACCIDENT_CATEGORY_DEFS = {
    "기계사고": {
        "triggers": ["기계사고", "기계 사고"],
        "patterns": ["기계", "기계기구", "기계·기구", "기계ㆍ기구", "프레스", "절단기", "전단기", "press"],
    },
    "전기사고": {
        "triggers": ["전기사고", "전기 사고", "감전사고", "감전 사고", "변압기 사고"],
        "patterns": ["전기", "감전", "변압기", "변전실", "고압", "누전", "전선", "분전반", "차단기"],
    },
    "교통사고": {
        "triggers": ["교통사고", "교통 사고"],
        "patterns": ["교통사고", "차량", "트럭", "화물차", "승용차", "도로", "주행", "운행중", "운행 중", "후진중", "후진 중"],
    },
    "화학물질 노출 사고": {
        "triggers": ["화학물질 노출 사고", "화학물질 노출사고", "화학물질노출사고", "화학물질 사고", "화학사고"],
        "patterns": [
            "화학물질", "화학", "유해물질", "유독물",
            "가스누출", "가스 누출", "유해가스", "유독가스",
            "중독", "흡입", "질식", "누출"
        ],
    },
    "크레인 사고": {
        "triggers": ["크레인사고", "크레인 사고"],
        "patterns": ["크레인", "천장크레인", "천정크레인", "이동식크레인", "이동식 크레인"],
    },
    "타워크레인 사고": {
        "triggers": ["타워크레인사고", "타워크레인 사고", "타워 크레인 사고"],
        "patterns": ["타워크레인", "타워 크레인"],
    },
    "굴삭기 사고": {
        "triggers": ["굴삭기사고", "굴삭기 사고", "포크레인 사고", "포클레인 사고"],
        "patterns": ["굴삭기", "포크레인", "포클레인", "백호", "excavator", "backhoe"],
    },
    "변압기 사고": {
        "triggers": ["변압기사고", "변압기 사고"],
        "patterns": ["변압기", "변전실", "변전소", "트랜스"],
    },
}


# -----------------------------
# 2) 필터/통계 관련 유틸
# -----------------------------
def parse_filters_from_question(question: str, known_types=None) -> Dict:
    import re

    q_raw = question
    q = question.replace(" ", "")
    filters = {}

    # 연도
    m = re.search(r"(\d{4})년", q)
    if m:
        filters["year"] = m.group(1)

    # 지역
    for r in REGION_CANDIDATES:
        if r in q:
            filters["region"] = r
            break

    # 사고유형
    types = []
    if known_types:
        for t in known_types:
            t_norm = str(t).strip()
            if t_norm and t_norm in q_raw:
                types.append(t_norm)

    common_type_words = [
        "붕괴", "추락", "협착", "끼임", "낙하",
        "충돌", "전도", "감전", "폭발", "화재", "질식", "중독"
    ]
    for w in common_type_words:
        if w in q_raw and w not in types:
            types.append(w)

    # 장비 키워드
    kw_list = []
    for kw in FORKLIFT_KEYWORDS:
        if kw in q_raw:
            kw_list.append(kw)

    # 상위 카테고리
    for cat_name, defs in ACCIDENT_CATEGORY_DEFS.items():
        triggers = defs.get("triggers", [])
        patterns = defs.get("patterns", [])
        if any(tr in q_raw for tr in triggers):
            types.extend(patterns)
            kw_list.extend(patterns)

    if types:
        filters["types"] = list(dict.fromkeys(types))
    if kw_list:
        filters["keywords"] = list(dict.fromkeys(kw_list))

    return filters


def doc_matches_filters(doc, filters: Dict) -> bool:
    if not filters:
        return True

    meta = getattr(doc, "metadata", {}) or {}
    text = normalize_str(getattr(doc, "page_content", ""))

    meta_occ_date = normalize_str(meta.get("occurrence_date", "") or meta.get("발생일자", ""))
    meta_region = normalize_str(meta.get("region_raw", "") or meta.get("지역", "") or meta.get("location", ""))
    meta_type = normalize_str(meta.get("accident_type", "") or meta.get("사고유형", ""))
    meta_keyword = normalize_str(meta.get("keyword", ""))
    meta_title = normalize_str(meta.get("title", ""))
    meta_contents = normalize_str(meta.get("contents", ""))
    meta_text_embed = normalize_str(meta.get("text_for_embedding", "") or meta.get("text", ""))

    # 연도
    year = filters.get("year")
    if year:
        year_ok = False
        if year in meta_occ_date or year in text or year in meta_contents or year in meta_text_embed:
            year_ok = True
        if not year_ok:
            return False

    # 지역
    region = filters.get("region")
    if region:
        region_ok = False
        if region in meta_region or region in text or region in meta_contents or region in meta_text_embed:
            region_ok = True
        if not region_ok:
            return False

    # 사고유형
    types = filters.get("types")
    if types:
        type_ok = False
        for t in types:
            if (
                t in meta_type
                or t in meta_keyword
                or t in meta_title
                or t in meta_contents
                or t in meta_text_embed
                or t in text
            ):
                type_ok = True
                break
        if not type_ok:
            return False

    # 장비 키워드
    keywords = filters.get("keywords")
    if keywords:
        kw_ok = False
        for kw in keywords:
            if (
                kw in meta_keyword
                or kw in meta_title
                or kw in meta_contents
                or kw in meta_text_embed
                or kw in text
            ):
                kw_ok = True
                break
        if not kw_ok:
            return False

    return True


def count_matching_accidents(filters: Dict, docs: List) -> int:
    if not filters:
        return 0

    accident_ids = set()
    no_id_count = 0

    for idx, d in enumerate(docs):
        if not doc_matches_filters(d, filters):
            continue

        meta = getattr(d, "metadata", {}) or {}
        acc_id = meta.get("accident_id")
        if acc_id:
            accident_ids.add(str(acc_id))
        else:
            no_id_count += 1

    if accident_ids:
        return len(accident_ids)
    else:
        return no_id_count


def compute_equipment_stats(docs: List, base_filters: Dict = None) -> Dict[str, int]:
    if base_filters is None:
        base_filters = {}
    else:
        base_filters = deepcopy(base_filters)
        if "keywords" in base_filters:
            base_filters.pop("keywords")

    equip_to_accidents = {label: set() for label in EQUIPMENT_CATEGORIES.keys()}

    for idx, d in enumerate(docs):
        if not doc_matches_filters(d, base_filters):
            continue

        meta = getattr(d, "metadata", {}) or {}
        acc_id = meta.get("accident_id")
        if acc_id:
            acc_key = str(acc_id)
        else:
            acc_key = f"doc_{idx}"

        text = normalize_str(getattr(d, "page_content", ""))
        meta_keyword = normalize_str(meta.get("keyword", ""))
        meta_title = normalize_str(meta.get("title", ""))
        meta_contents = normalize_str(meta.get("contents", ""))
        meta_text_embed = normalize_str(meta.get("text_for_embedding", "") or meta.get("text", ""))

        combined = " ".join([text, meta_keyword, meta_title, meta_contents, meta_text_embed])

        for label, patterns in EQUIPMENT_CATEGORIES.items():
            for p in patterns:
                if p and p in combined:
                    equip_to_accidents[label].add(acc_key)
                    break

    result = {label: len(acc_ids) for label, acc_ids in equip_to_accidents.items() if acc_ids}
    return result


# -----------------------------
# 3) 벡터스토어 로딩 + 통계 생성
# -----------------------------
def load_vectorstore(
    embedding_model: str,
    index_path: str = "faiss_index"
) -> Tuple[FAISS, List]:
    """
    Streamlit에 의존하지 않는 순수 로딩 함수.
    """
    store = FAISS.load_local(
        folder_path=index_path,
        embeddings=OpenAIEmbeddings(model=embedding_model),
        allow_dangerous_deserialization=True,
    )

    docstore = store.docstore
    if hasattr(docstore, "_dict"):
        docs = list(docstore._dict.values())
    else:
        docs = []

    return store, docs


def build_accident_stats(docs: List) -> Dict:
    """
    기존 load_vectorstore_and_stats 안에서 하던 통계 계산만 분리
    """
    total_docs = len(docs)
    accident_ids = set()
    accident_type_labels = set()

    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        acc_id = meta.get("accident_id")
        if acc_id:
            accident_ids.add(str(acc_id))

        acc_type = meta.get("accident_type")
        if acc_type:
            accident_type_labels.add(str(acc_type).strip())

    if accident_ids:
        total_accidents = len(accident_ids)
    else:
        total_accidents = total_docs

    stats = {
        "total_docs": total_docs,
        "total_accidents": total_accidents,
        "accident_type_labels": sorted(accident_type_labels),
    }
    return stats


def answer_with_stats_using_index(
    question: str,
    stats: Dict,
    docs: List,
) -> str | None:
    """
    기존 try_answer_with_stats_using_index 그대로 옮겨둔 버전.
    통계 질문이 아니면 None 리턴.
    """
    q_raw = question
    q = question.replace(" ", "").strip()

    is_count_query = (
        ("사고" in q or "사례" in q or "재해" in q)
        and (
            "건수" in q or "몇건" in q or "몇개" in q or
            "몇건이야" in q or "몇개야" in q or "얼마나" in q or "통계" in q
        )
    )

    if not is_count_query:
        return None

    known_types = stats.get("accident_type_labels", [])
    filters = parse_filters_from_question(q_raw, known_types=known_types)

    # 장비별 통계?
    is_equipment_summary_query = (
        ("장비" in q or "설비" in q)
        and ("별" in q or "종류" in q or "분류" in q or "통계" in q)
    )

    if is_equipment_summary_query:
        base_filters = {k: v for k, v in filters.items() if k in ("year", "region", "types")}
        equip_stats = compute_equipment_stats(docs, base_filters=base_filters)

        if not equip_stats:
            return "현재 조건에 해당하는 장비별 사고 사례를 인덱스에서 찾기 어렵습니다."

        sorted_items = sorted(equip_stats.items(), key=lambda x: x[1], reverse=True)

        lines = []
        for label, cnt in sorted_items[:8]:
            lines.append(f"- {label}: {cnt:,}건")

        cond_parts = []
        if "year" in base_filters:
            cond_parts.append(f"{base_filters['year']}년")
        if "region" in base_filters:
            cond_parts.append(base_filters["region"])
        if "types" in base_filters:
            cond_parts.append("/".join(base_filters["types"]) + " 사고")

        if cond_parts:
            cond_text = " / ".join(cond_parts)
            title = f"현재 벡터 인덱스 기준, **{cond_text} 장비별 사고 건수**는 다음과 같습니다:\n\n"
        else:
            title = "현재 벡터 인덱스 기준, **장비별 사고 건수**는 대략 다음과 같습니다:\n\n"

        return title + "\n".join(lines)

    # 일반 건수 질문
    if not filters:
        total_acc = stats.get("total_accidents")
        total_docs = stats.get("total_docs")
        return (
            f"현재 이 챗봇이 참조하는 벡터 인덱스 기준으로는 "
            f"**총 {total_acc:,}건의 사고 사례**가 있습니다.\n\n"
            f"(인덱스에는 총 {total_docs:,}개의 문서가 저장되어 있으며, "
            f"여러 문서가 하나의 사고 사례에 대응될 수 있습니다.)"
        )

    count = count_matching_accidents(filters, docs)

    parts = []
    if "year" in filters:
        parts.append(f"{filters['year']}년")
    if "region" in filters:
        parts.append(filters["region"])
    if "types" in filters:
        type_label = "/".join(filters["types"])
        parts.append(f"{type_label} 사고")
    if "keywords" in filters:
        kw_label = "/".join(filters["keywords"])
        parts.append(f"{kw_label} 관련")

    if not parts:
        prefix = "해당 조건"
    else:
        prefix = " ".join(parts)

    return (
        f"현재 벡터 인덱스 기준으로, **{prefix} 사고 사례는 약 {count:,}건**입니다.\n\n"
        f"(이 숫자는 인덱스에 포함된 데이터 기준이며, 원본 전체 데이터 건수와는 차이가 있을 수 있습니다.)"
    )


# -----------------------------
# 4) RAG 검색 관련 유틸
# -----------------------------
def build_metadata_filter(query: str) -> Dict:
    """
    원래 build_metadata_filter 그대로
    """
    filters = {}

    accident_keywords = {
        "감전": "감전", "전기": "감전",
        "화재": "화재",
        "추락": "추락",
        "협착": "협착",
        "질식": "질식",
        "중독": "중독", "가스": "중독",
    }

    for k, v in accident_keywords.items():
        if k in query:
            filters["accident_type"] = v

    if any(w in query for w in ["응급", "조치", "대처"]):
        filters["section"] = "응급조치"
    if "예방" in query:
        filters["section"] = "예방"
    if any(w in query for w in ["법", "규정"]):
        filters["section"] = "법규"

    return filters


def format_docs(docs: List) -> str:
    formatted = []
    for i, d in enumerate(docs):
        md = d.metadata
        header = (
            f"[문서 {i+1}] "
            f"(type: {md.get('accident_type','N/A')}, "
            f"section: {md.get('section','N/A')}, "
            f"source: {md.get('source','N/A')}, "
            f"page: {md.get('page_num','?')})"
        )
        formatted.append(header + "\n" + d.page_content)
    return "\n\n".join(formatted)


def create_rag_chain(
    vectorstore,
    rag_prompt_path: str = "prompts/first.yaml",
    llm_model_name: str = "gpt-4o-mini",
):
    """
    원래 create_rag_chain 그대로 옮기되,
    model 이름/프롬프트 경로를 인자로 받을 수 있게 해둠.
    """
    rag_prompt = load_prompt(rag_prompt_path, encoding="utf-8")
    llm = ChatOpenAI(model=llm_model_name, temperature=0)

    condense_prompt = ChatPromptTemplate.from_template(
        """
        너는 검색용 질문을 재작성하는 보조 도우미이다.

        아래 대화 이력을 참고하여, 사용자의 마지막 질문을
        검색 엔진에 넣을 **한 개의 독립적인 질문**으로 바꿔라.

        - 원래 질문에 등장하는 핵심 키워드(장비명, 법규명, 사고유형 등)는 절대 삭제하지 마라.
        - '그 사고', '이 사례', '위에서 말한 것' 같은 표현은
          대화 이력에 등장한 구체적인 내용으로 풀어써라.
        - 새로운 사실을 지어내지 말고, 대화에 나온 정보만 사용해라.

        [대화 이력]
        {history}

        [사용자의 마지막 질문]
        {question}

        [재작성된 검색용 질문]
        """
    )
    condense_chain = condense_prompt | llm | StrOutputParser()

    def retrieve_with_dual_search(inputs):
        query = inputs["question"]
        history = inputs.get("history", "")

        metadata_filter = build_metadata_filter(query)

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.4,
                "filter": metadata_filter if metadata_filter else None,
            },
        )

        docs_original = retriever.invoke(query)

        docs_contextual = []
        if history.strip():
            rewritten = condense_chain.invoke(
                {"history": history, "question": query}
            ).strip()
            if rewritten:
                try:
                    docs_contextual = retriever.invoke(rewritten)
                except Exception:
                    docs_contextual = []

        seen = set()
        final_docs = []
        for d in docs_original + docs_contextual:
            content = d.page_content
            if content not in seen:
                seen.add(content)
                final_docs.append(d)

        return format_docs(final_docs)

    chain = (
        RunnableMap(
            {
                "context": RunnableLambda(retrieve_with_dual_search),
                "question": RunnableLambda(lambda x: x["question"]),
                "history": RunnableLambda(lambda x: x["history"]),
            }
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain
