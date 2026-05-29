"""
담론-현실 간극지수 (RQ4a) — Validity Guardrail #1·#2
====================================================
뉴스(담론)와 국내공고(현실)에서 '기술/역량 사전 토큰'만 비교한다(프레임어 제외).
- Spearman 순위상관 ρ: 전체 일치도
- 순위차 Δrank: 사분면 분류 (담론 과잉 / 조용한 핵심 / 공통 강조)
"""

import pandas as pd
from scipy.stats import spearmanr


def skill_frequency(skill_lists):
    """[[skill,...], ...] → 상대빈도 Series(내림차순)."""
    from collections import Counter
    c = Counter(s for lst in skill_lists for s in (lst or []))
    total = sum(c.values()) or 1
    return pd.Series({k: v / total for k, v in c.items()}).sort_values(ascending=False)


def gap_table(news_freq, job_freq, top_n=40):
    """공통 토큰의 순위·순위차·사분면 라벨 + Spearman ρ 반환."""
    common = sorted(set(news_freq.index) | set(job_freq.index))
    df = pd.DataFrame(index=common)
    df["news_freq"] = news_freq.reindex(common).fillna(0)
    df["job_freq"] = job_freq.reindex(common).fillna(0)
    # 순위(낮을수록 상위)
    df["news_rank"] = df["news_freq"].rank(ascending=False, method="min")
    df["job_rank"] = df["job_freq"].rank(ascending=False, method="min")
    df["delta_rank"] = df["news_rank"] - df["job_rank"]  # +면 공고에서 더 상위(조용한 핵심)

    def label(row):
        if row["delta_rank"] >= 10:
            return "조용한 핵심(공고>뉴스)"
        if row["delta_rank"] <= -10:
            return "담론 과잉(뉴스>공고)"
        return "공통"
    df["quadrant"] = df.apply(label, axis=1)

    # Spearman: 양쪽에 등장한 토큰만
    both = df[(df["news_freq"] > 0) & (df["job_freq"] > 0)]
    rho, p = spearmanr(both["news_rank"], both["job_rank"]) if len(both) > 2 else (float("nan"), float("nan"))
    df = df.sort_values("job_freq", ascending=False).head(top_n)
    return df, rho, p
