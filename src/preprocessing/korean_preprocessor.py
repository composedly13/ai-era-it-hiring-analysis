"""
한국어 텍스트 전처리 (워크넷·뉴스)
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


def extract_nouns(text: str, tagger=None) -> list[str]:
    """KoNLPy tagger로 명사 추출 후 불용어 제거."""
    # TODO: 구현 — tagger는 외부에서 주입(Okt / Mecab)
    raise NotImplementedError


def preprocess(text: str, tagger=None) -> list[str]:
    """전체 전처리 파이프라인."""
    text = remove_html(text)
    # TODO: 특수문자 정제, 명사 추출, 불용어 제거
    raise NotImplementedError
