"""
한국어 텍스트 전처리 (뉴스·국내공고)

KoNLPy(Okt) 기반:
  - 한국어 명사(Noun) 추출
  - 영문 토큰(Alpha) 동시 보존 — 기술스택(Python·AWS·ChatGPT 등) 손실 방지
  - HTML·특수문자 정제, 불용어 제거, 1글자 토큰 제외

기술스택 추출은 별도 — src.preprocessing.tech_dictionary.extract_skills 사용.
이 모듈은 워드클라우드·일반 명사 빈도·토픽 모델링 입력용.
"""

import re


# 채용공고 + 뉴스 공통 불용어
STOPWORDS = {
    # 채용공고 메타
    "모집", "담당", "업무", "자격", "우대", "근무", "지원",
    "회사", "가능자", "이상", "보유", "필요", "관련", "사용",
    "운영", "관리",
    # 뉴스 메타 (저작권·서지)
    "기자", "사진", "무단", "전재", "재배포", "금지", "기사",
    "보도", "제공", "독자", "여러분", "제보", "메일", "전화",
    "카카오톡", "광주", "전남", "지난해", "올해", "최근", "현재",
    "지난", "이번", "이날", "이런", "통해", "대해", "위해",
    # 일반 빈출 — 정보량 거의 없음
    "년", "월", "일", "때", "후", "전", "중", "위", "곳", "데",
    "건", "명", "개", "원", "수", "것", "등", "다음", "이상",
    "이하", "이내", "정도", "가량",
}


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_CLEAN_RE = re.compile(r"[^가-힣A-Za-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """HTML·특수문자 정제."""
    if not isinstance(text, str) or not text:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _CLEAN_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def extract_tokens(text: str, tagger=None, min_len: int = 2) -> list[str]:
    """명사(Noun) + 영문(Alpha) 토큰 추출.

    Parameters
    ----------
    text : str
    tagger : konlpy.tag.Okt | None
        Okt 인스턴스 (재사용 권장). None이면 모듈 싱글톤 사용.
    min_len : int
        토큰 최소 길이 (1글자 제외)

    Returns
    -------
    list[str]
        중복 보존 — 빈도 계산용
    """
    if not text:
        return []
    if tagger is None:
        tagger = _get_okt()

    tokens = []
    for word, pos in tagger.pos(text):
        if pos not in ("Noun", "Alpha"):
            continue
        if len(word) < min_len:
            continue
        if word in STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def preprocess(text: str, tagger=None) -> list[str]:
    """clean_text → extract_tokens."""
    return extract_tokens(clean_text(text), tagger=tagger)


# ─── 내부 ─────────────────────────────────────────────────────────
_okt_singleton = None


def _get_okt():
    """Okt 싱글톤 — JVM 부팅 비용 회피."""
    global _okt_singleton
    if _okt_singleton is None:
        from konlpy.tag import Okt
        _okt_singleton = Okt()
    return _okt_singleton
