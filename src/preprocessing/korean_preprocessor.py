"""
한국어 텍스트 전처리 (국내공고·뉴스)
KoNLPy 형태소 분석 기반 명사·영문·숫자 조합 토큰 추출
"""

import re
import pandas as pd

# 채용공고 특화 불용어
JOB_STOPWORDS = {
    "모집", "담당", "업무", "자격", "우대", "근무", "지원", "채용",
    "회사", "가능자", "이상", "경험", "보유", "필요", "관련", "사용",
    "개발", "서비스", "구현", "운영", "관리", "설계", "분석",
}


def remove_html(text: str) -> str:
    """HTML 태그 제거."""
    return re.sub(r"<[^>]+>", " ", text)


def clean_text(text: str) -> str:
    """HTML 제거 + 특수문자 정리. 기술명 보존 위해 +, #, . 은 남긴다."""
    text = remove_html(text)
    text = re.sub(r"[^0-9A-Za-z가-힣\+\#\.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _ensure_tagger(tagger):
    if tagger is None:
        from konlpy.tag import Okt
        return Okt()
    return tagger


def extract_nouns(text: str, tagger=None) -> list[str]:
    """KoNLPy tagger로 한글 명사 추출 후 불용어·1글자 제거."""
    if not isinstance(text, str) or not text.strip():
        return []
    tagger = _ensure_tagger(tagger)
    nouns = tagger.nouns(clean_text(text))
    return [n for n in nouns if len(n) > 1 and n not in JOB_STOPWORDS]


def preprocess(text: str, tagger=None) -> list[str]:
    """전체 전처리 — 한글 명사 + 영문 기술토큰(AI·Python 등) 보존.

    Okt.nouns는 영문을 명사로 잡지 않으므로 영문 토큰은 정규식으로 별도 추출한다.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = clean_text(text)
    tagger = _ensure_tagger(tagger)
    nouns = [n for n in tagger.nouns(cleaned) if len(n) > 1]
    eng = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.]{1,}", cleaned)
    toks = nouns + eng
    return [t for t in toks if t not in JOB_STOPWORDS]
