"""
시계열 분석 — 뉴스 담론 변화 (RQ3·RQ4)
========================================
역할 분리 (Validity Guardrail #4):
  뉴스       : 장기 시계열 (2021–2026, before/after ChatGPT)
  LinkedIn   : 단기 단면 (2024-04)
  국내공고   : 현재 단면 (2026-05)

01 노트북 산출물 (data/processed/news/news_processed.csv) 스키마 가정:
  date(YYYY-MM-DD str), year(int), month('YYYY-MM' str), period('before'/'after'),
  skills(';' 구분 str), tier_a~d_*(0/1), title_tokens(' ' 구분), body_tokens(' ' 구분)
"""

import pandas as pd
from collections import Counter


# ChatGPT 출시일 (담론 단절점)
CHATGPT_LAUNCH = pd.Timestamp("2022-11-30")


# ─── 기사 수 ─────────────────────────────────────────────────────────
def monthly_count(df: pd.DataFrame, month_col: str = "month") -> pd.Series:
    """월별 총 기사 수 Series. index='YYYY-MM' (str)."""
    s = df[month_col].value_counts().sort_index()
    s.name = "n_articles"
    return s


def yearly_count(df: pd.DataFrame, year_col: str = "year") -> pd.Series:
    """연도별 기사 수."""
    s = df[year_col].value_counts().sort_index()
    s.name = "n_articles"
    return s


# ─── 키워드 / Tier 시계열 ────────────────────────────────────────────
def _parse_skills(value) -> set:
    """';' 구분 문자열 → set."""
    if not isinstance(value, str) or not value:
        return set()
    return {s for s in value.split(";") if s}


def monthly_skill_mention(df: pd.DataFrame, skill: str,
                          skills_col: str = "skills",
                          month_col: str = "month") -> pd.Series:
    """월별 특정 canonical 스킬을 포함한 기사 수."""
    if "_skills_set" not in df.columns:
        df = df.copy()
        df["_skills_set"] = df[skills_col].apply(_parse_skills)
    has = df["_skills_set"].apply(lambda s: skill in s)
    s = df.loc[has, month_col].value_counts().sort_index()
    s.name = skill
    return s


def monthly_skill_rate(df: pd.DataFrame, skill: str,
                       skills_col: str = "skills",
                       month_col: str = "month") -> pd.Series:
    """월별 특정 스킬 언급률(%) — 분모: 그 달 전체 기사."""
    mention = monthly_skill_mention(df, skill, skills_col, month_col)
    total = monthly_count(df, month_col)
    rate = (mention.reindex(total.index, fill_value=0) / total * 100)
    rate.name = f"{skill}_rate"
    return rate


def monthly_tier_share(df: pd.DataFrame, tier_col: str,
                       month_col: str = "month") -> pd.Series:
    """월별 특정 AI tier 보유율(%)."""
    g = df.groupby(month_col)[tier_col].mean() * 100
    g.name = tier_col
    return g.sort_index()


def yearly_tier_share(df: pd.DataFrame, tier_col: str,
                      year_col: str = "year") -> pd.Series:
    """연도별 AI tier 보유율(%) — 월별 노이즈 줄임."""
    g = df.groupby(year_col)[tier_col].mean() * 100
    g.name = tier_col
    return g.sort_index()


# ─── before/after ─────────────────────────────────────────────────────
def before_after_split(df: pd.DataFrame, period_col: str = "period") -> dict:
    """period 컬럼 기준 before/after 분리."""
    return {
        "before": df[df[period_col] == "before"],
        "after":  df[df[period_col] == "after"],
    }


def period_token_freq(df: pd.DataFrame, period: str,
                      token_col: str = "body_tokens",
                      period_col: str = "period",
                      top_n: int = 100) -> pd.Series:
    """period(before/after)별 토큰 빈도 top_n."""
    sub = df[df[period_col] == period]
    counter: Counter = Counter()
    for toks in sub[token_col].fillna(""):
        if toks:
            counter.update(toks.split())
    s = pd.Series(dict(counter.most_common(top_n)))
    s.name = f"{period}_top{top_n}"
    return s


# ─── 프레임(담론) ────────────────────────────────────────────────────
# RQ4b: 위기/대체/재편 프레임어 사전
FRAMING_KEYWORDS = {
    "crisis": [   # 위기·불안
        "위기", "한파", "감축", "축소", "줄이", "줄어", "감소",
        "불안", "공포", "암울", "어렵", "어려운", "어려워",
    ],
    "replace": [  # 대체·소멸
        "대체", "사라지", "사라질", "없어지", "사라진",
        "잡포칼립스", "실직", "해고",
    ],
    "restructure": [  # 재편·변화
        "재편", "변화", "전환", "혁신", "패러다임",
        "달라지", "달라진", "새로운", "전면", "리스킬",
    ],
    "opportunity": [  # 기회·성장
        "기회", "성장", "확대", "증가", "늘어", "늘리",
        "활성화", "활약", "수요",
    ],
}


def count_frame_mentions(text: str, frame_keywords: list) -> int:
    """텍스트 내 프레임 키워드 등장 횟수 합 (단순 substring)."""
    if not isinstance(text, str) or not text:
        return 0
    return sum(text.count(kw) for kw in frame_keywords)


def monthly_frame_intensity(df: pd.DataFrame, frame: str,
                            text_col: str = "body",
                            month_col: str = "month") -> pd.Series:
    """월별 프레임 강도(기사당 평균 등장 횟수)."""
    keywords = FRAMING_KEYWORDS[frame]
    if f"_frame_{frame}" not in df.columns:
        df = df.copy()
        df[f"_frame_{frame}"] = df[text_col].fillna("").apply(
            lambda t: count_frame_mentions(t, keywords)
        )
    g = df.groupby(month_col)[f"_frame_{frame}"].mean()
    g.name = frame
    return g.sort_index()
