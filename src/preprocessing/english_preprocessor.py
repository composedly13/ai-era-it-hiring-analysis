"""
영어 텍스트 전처리 (LinkedIn)
lemmatization, stopwords 제거, job title 정규화
"""

import re
import pandas as pd


def normalize_title(title: str) -> str:
    """job title 정규화 (대소문자·약어 통일)."""
    # TODO: 구현
    raise NotImplementedError


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """NLTK/spaCy lemmatization."""
    # TODO: 구현
    raise NotImplementedError


def preprocess(text: str) -> list[str]:
    """전체 영어 전처리 파이프라인."""
    # TODO: 구현
    raise NotImplementedError
