"""
04_eda_keyword_analysis.py
담당: 미정

탐색적 분석 — 빈도·TF-IDF·WordCloud·시계열
RQ1(국내공고 상위 역량) / RQ2(LinkedIn AI 역량) / RQ3(뉴스 담론 변화)
출력: outputs/figures/
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ast
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from src.analysis.frequency_analysis import token_frequency
from src.analysis.tfidf_analysis import top_terms_per_group
from src.analysis.time_series import monthly_keyword_count
from src.visualization.wordcloud_gen import generate_wordcloud

KR_PATH  = "data/processed/kr_jobs_clean.csv"
NEWS_PATH     = "data/processed/news/news_processed.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"

os.makedirs(FIG_DIR, exist_ok=True)


def load_token_lists(df: pd.DataFrame, col: str = "tech_tokens") -> list[list[str]]:
    return [ast.literal_eval(r) if isinstance(r, str) else r
            for r in df[col].fillna("[]")]


# ---------------------------------------------------------------------------
# RQ1. 국내공고 상위 기술스택 빈도
# ---------------------------------------------------------------------------
def rq1_kr_jobs_top_skills() -> None:
    df = pd.read_csv(KR_PATH)
    freq = token_frequency(load_token_lists(df))

    top20 = freq.head(20)
    top20[::-1].plot(kind="barh", figsize=(10, 7), color="#dd8452")
    plt.title("RQ1: 국내 IT 채용공고 상위 요구 기술스택 (국내공고)", fontsize=13)
    plt.xlabel("언급 공고 수")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_jobs_top_skills.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[RQ1] 저장:", f"{FIG_DIR}/kr_jobs_top_skills.png")


# ---------------------------------------------------------------------------
# RQ2. LinkedIn AI 역량 비율
# ---------------------------------------------------------------------------
def rq2_linkedin_ai_skills() -> None:
    df = pd.read_csv(LINKEDIN_PATH)
    freq = token_frequency(load_token_lists(df))

    top20 = freq.head(20)
    top20[::-1].plot(kind="barh", figsize=(10, 7), color="#4c72b0")
    plt.title("RQ2: 글로벌 IT 채용공고 상위 기술스택 (LinkedIn 2023–24)", fontsize=13)
    plt.xlabel("언급 공고 수")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/linkedin_top_skills.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[RQ2] 저장:", f"{FIG_DIR}/linkedin_top_skills.png")


# ---------------------------------------------------------------------------
# RQ3. 뉴스 AI 키워드 월별 언급률 시계열
# ---------------------------------------------------------------------------
def rq3_news_timeseries() -> None:
    df = pd.read_csv(NEWS_PATH, parse_dates=["발행일"])
    df_monthly = df.set_index("발행일").resample("M").size().rename("total")

    ai_series = monthly_keyword_count(df, date_col="발행일", keyword_col="text", keyword="AI")
    ratio = (ai_series / df_monthly * 100).fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    df_monthly.plot(ax=axes[0], title="월별 뉴스 기사 수 (IT 채용 관련)", color="steelblue")
    ratio.plot(ax=axes[1], title="AI 키워드 월별 언급률 (%)", color="tomato")
    axes[1].axvline("2022-11-01", color="gray", linestyle="--", label="ChatGPT 출시")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_ai_trend.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[RQ3] 저장:", f"{FIG_DIR}/news_ai_trend.png")


# ---------------------------------------------------------------------------
# WordCloud — 국내공고 / 뉴스 before·after
# ---------------------------------------------------------------------------
def wordcloud_all() -> None:
    df_w = pd.read_csv(KR_PATH)
    freq_w = token_frequency(load_token_lists(df_w)).to_dict()
    generate_wordcloud(freq_w, title="국내공고 기술스택", save_path=f"{FIG_DIR}/wc_kr_jobs.png")

    df_n = pd.read_csv(NEWS_PATH)
    for period in ["before", "after"]:
        sub = df_n[df_n["period"] == period]
        freq = token_frequency(load_token_lists(sub)).to_dict()
        generate_wordcloud(freq, title=f"뉴스 기술 키워드 ({period} ChatGPT)",
                           save_path=f"{FIG_DIR}/wc_news_{period}.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rq1_kr_jobs_top_skills()
    rq2_linkedin_ai_skills()
    rq3_news_timeseries()
    wordcloud_all()
