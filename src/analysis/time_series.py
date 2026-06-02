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
    """월별, keyword_col(파이프 구분 canonical 토큰)에 keyword 토큰을 포함한 기사 수.

    substring이 아닌 정확 토큰 매칭 — "AI"가 영어 단어 속 "ai"(email·train 등)에
    오탐되는 것을 방지. keyword_col 은 extract_skills 결과("AI|Python|...") 가정.
    """
    s = df[[date_col, keyword_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    mask = s[keyword_col].apply(
        lambda v: isinstance(v, str) and keyword in v.split("|")
    )
    return s[mask].dropna(subset=[date_col]).set_index(date_col).resample("M").size()


def before_after_comparison(df: pd.DataFrame, date_col: str, cutoff: str = CHATGPT_LAUNCH) -> dict:
    """cutoff 기준 before/after 그룹 분리."""
    before = df[df[date_col] < cutoff]
    after  = df[df[date_col] >= cutoff]
    return {"before": before, "after": after}
