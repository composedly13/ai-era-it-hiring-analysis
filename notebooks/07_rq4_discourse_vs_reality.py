"""
07_rq4_discourse_vs_reality.py
담당: A·B·D (공동)

핵심 분석 — RQ4(담론 vs 현실 간극) + RQ5(국내 vs 글로벌 격차)
출력: outputs/figures/rq4_*.png, outputs/rq4_gap_table.csv

Validity Guardrail:
  #1 공정 비교 — 동일 시점(2026 뉴스 vs 2026 국내공고)
  #2 비교 어휘 분리 — RQ4a 기술/역량 사전 토큰만 / 담론 프레임(RQ4b)은 별도(08·time_series)
  #3 한국어↔영어 토픽 직접 비교 금지 — 기술스택 사전 토큰 빈도만 비교
"""

import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["axes.unicode_minus"] = False
# 한글 폰트: macOS=AppleGothic / Windows=Malgun Gothic
for _f in ("AppleGothic", "Malgun Gothic", "NanumGothic"):
    try:
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        pass

from src.analysis.gap_index import skill_frequency, gap_table

KR_PATH = "data/processed/domestic/kr_jobs_clean.jsonl"  # 국내공고(점핏+원티드)
NEWS_PATH = "data/processed/news/news_processed.csv"      # 뉴스(skills + 발행일)
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"  # 글로벌(skills)
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_kr_skill_freq():
    rows = [json.loads(l) for l in open(KR_PATH, encoding="utf-8")]
    return skill_frequency([r.get("skills", []) for r in rows])


def load_news_skill_freq_2026():
    if not os.path.exists(NEWS_PATH):
        print(f"  [skip] {NEWS_PATH} 없음 — 뉴스 수집(01) 후 실행")
        return None
    df = pd.read_csv(NEWS_PATH)
    if "발행일" in df.columns:
        df = df[pd.to_datetime(df["발행일"], errors="coerce").dt.year == 2026]
    lists = [json.loads(s) if isinstance(s, str) and s.startswith("[") else []
             for s in df.get("skills", pd.Series(dtype=str)).fillna("[]")]
    return skill_frequency(lists)


def load_linkedin_skill_freq():
    if not os.path.exists(LINKEDIN_PATH):
        print(f"  [skip] {LINKEDIN_PATH} 없음 — LinkedIn 처리(02) 후 실행")
        return None
    df = pd.read_csv(LINKEDIN_PATH)
    lists = [json.loads(s) if isinstance(s, str) and s.startswith("[") else []
             for s in df.get("skills", pd.Series(dtype=str)).fillna("[]")]
    return skill_frequency(lists)


def rq4_gap_index(news_freq, kr_freq):
    """RQ4a: 담론(뉴스) vs 현실(국내공고) 기술스택 간극지수."""
    df, rho, p = gap_table(news_freq, kr_freq, top_n=40)
    df.to_csv("outputs/rq4_gap_table.csv", encoding="utf-8-sig")
    print(f"  [RQ4a] Spearman ρ = {rho:.3f} (p={p:.3g})")
    print("  담론 과잉:", list(df[df.quadrant.str.startswith('담론')].index[:8]))
    print("  조용한 핵심:", list(df[df.quadrant.str.startswith('조용')].index[:8]))
    # 사분면 산점도
    plt.figure(figsize=(9, 9))
    plt.scatter(df["news_rank"], df["job_rank"], alpha=0.6)
    for sk, r in df.iterrows():
        plt.annotate(sk, (r["news_rank"], r["job_rank"]), fontsize=8)
    lim = max(df[["news_rank", "job_rank"]].max()) + 2
    plt.plot([0, lim], [0, lim], "--", color="gray")
    plt.xlabel("뉴스 담론 순위 →"); plt.ylabel("국내공고 순위 →")
    plt.title(f"RQ4a 담론 vs 현실 간극 (Spearman ρ={rho:.2f})")
    plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/rq4_gap_index.png", dpi=150)
    print(f"  저장: {FIG_DIR}/rq4_gap_index.png")


def rq5_domestic_vs_global(kr_freq, linkedin_freq, top=20):
    """RQ5: 국내(점핏·원티드) vs 글로벌(LinkedIn) 기술스택 비중."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    k = kr_freq.head(top); l = linkedin_freq.head(top)
    axes[0].barh(k.index[::-1], k.values[::-1], color="#dd8452")
    axes[0].set_title("국내 실제 요구역량 (점핏·원티드)")
    axes[1].barh(l.index[::-1], l.values[::-1], color="#4c72b0")
    axes[1].set_title("글로벌 채용공고 (LinkedIn 2023–24)")
    plt.suptitle("RQ5 국내 vs 글로벌 기술스택 (기술사전 단위)")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/rq5_domestic_vs_global.png", dpi=150)
    print(f"  저장: {FIG_DIR}/rq5_domestic_vs_global.png")


if __name__ == "__main__":
    kr_freq = load_kr_skill_freq()
    print(f"[국내공고] 상위: {list(kr_freq.head(10).index)}")
    news_freq = load_news_skill_freq_2026()
    if news_freq is not None:
        rq4_gap_index(news_freq, kr_freq)
    linkedin_freq = load_linkedin_skill_freq()
    if linkedin_freq is not None:
        rq5_domestic_vs_global(kr_freq, linkedin_freq)
