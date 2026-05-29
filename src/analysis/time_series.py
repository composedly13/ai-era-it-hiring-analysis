"""
시계열 분석 — 역할 분리 원칙 적용 (Validity Guardrail #4)
뉴스: 장기 시계열 (2021–2026, before/after AI)
LinkedIn: 단기 단면 (2023–2024 드리프트)
국내공고: 현재 단면
"""

import pandas as pd


# ChatGPT 출시 기준일
CHATGPT_LAUNCH = "2022-11-01"


def monthly_keyword_count(df: pd.DataFrame, date_col: str, keyword_col: str, keyword: str) -> pd.Series:
    """월별 특정 키워드 언급 건수 반환."""
    # TODO: 구현
    raise NotImplementedError


def before_after_comparison(df: pd.DataFrame, date_col: str, cutoff: str = CHATGPT_LAUNCH) -> dict:
    """cutoff 기준 before/after 그룹 분리."""
    before = df[df[date_col] < cutoff]
    after  = df[df[date_col] >= cutoff]
    return {"before": before, "after": after}
