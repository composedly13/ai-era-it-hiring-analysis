"""
07_rq4_discourse_vs_reality.py
담당: 미정

핵심 분석 — 담론(뉴스) vs 현실(워크넷) vs 글로벌(LinkedIn) 비교 (RQ4)
출력: outputs/figures/rq4_*.png

Validity Guardrail #1 — 공정한 비교 축:
  - 동일 시점: 2026년 뉴스 vs 2026년 워크넷 공고
  - 비교 단위: 기술스택 사전 토큰 (언어 무관 빈도/비율)
  - 한국어↔영어 토픽 결과 직접 비교 금지
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
from src.visualization.comparison_chart import plot_discourse_vs_reality

WORKNET_PATH  = "data/processed/worknet/worknet_processed.csv"
NEWS_PATH     = "data/processed/news/news_processed.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"

os.makedirs(FIG_DIR, exist_ok=True)


def load_tech_freq(path: str, col: str = "tech_tokens"):
    df = pd.read_csv(path)
    token_lists = [ast.literal_eval(r) if isinstance(r, str) else r
                   for r in df[col].fillna("[]")]
    return token_frequency(token_lists), df


# ---------------------------------------------------------------------------
# Step 1. 동일 시점 필터 (2026년)
# ---------------------------------------------------------------------------
def step1_same_period_news() -> pd.DataFrame:
    df = pd.read_csv(NEWS_PATH, parse_dates=["발행일"])
    df_2026 = df[df["발행일"].dt.year == 2026]
    print(f"[Step1] 2026년 뉴스: {len(df_2026):,}건")
    return df_2026


# ---------------------------------------------------------------------------
# Step 2. 3개 데이터소스 기술스택 빈도 계산
# ---------------------------------------------------------------------------
def step2_compute_frequencies(df_news_2026: pd.DataFrame):
    # 뉴스 2026년분
    token_lists_news = [ast.literal_eval(r) if isinstance(r, str) else r
                        for r in df_news_2026["tech_tokens"].fillna("[]")]
    news_freq = token_frequency(token_lists_news)

    # 워크넷 (현재 전체 = 2026 단면)
    worknet_freq, _ = load_tech_freq(WORKNET_PATH)

    # LinkedIn (글로벌 단면)
    linkedin_freq, _ = load_tech_freq(LINKEDIN_PATH)

    return news_freq, worknet_freq, linkedin_freq


# ---------------------------------------------------------------------------
# Step 3. 담론 vs 현실 비교 차트
# ---------------------------------------------------------------------------
def step3_discourse_vs_reality(news_freq, worknet_freq) -> None:
    plot_discourse_vs_reality(
        news_freq, worknet_freq, top_n=20,
        save_path=f"{FIG_DIR}/rq4_discourse_vs_reality.png"
    )
    print("[Step3] 저장:", f"{FIG_DIR}/rq4_discourse_vs_reality.png")


# ---------------------------------------------------------------------------
# Step 4. 국내 vs 글로벌 비교 차트
# ---------------------------------------------------------------------------
def step4_domestic_vs_global(worknet_freq, linkedin_freq) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    top = 20

    w_top = worknet_freq.head(top)
    l_top = linkedin_freq.head(top)

    axes[0].barh(w_top.index[::-1], w_top.values[::-1], color="#dd8452")
    axes[0].set_title("국내 실제 요구역량 (워크넷)", fontsize=12)

    axes[1].barh(l_top.index[::-1], l_top.values[::-1], color="#4c72b0")
    axes[1].set_title("글로벌 채용공고 (LinkedIn 2023–24)", fontsize=12)

    plt.suptitle("국내 vs 글로벌 기술스택 비교 (기술사전 단위)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq4_domestic_vs_global.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Step4] 저장:", f"{FIG_DIR}/rq4_domestic_vs_global.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_news_2026 = step1_same_period_news()
    news_freq, worknet_freq, linkedin_freq = step2_compute_frequencies(df_news_2026)
    step3_discourse_vs_reality(news_freq, worknet_freq)
    step4_domestic_vs_global(worknet_freq, linkedin_freq)
