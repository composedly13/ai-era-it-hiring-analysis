"""
07_rq4_discourse_vs_reality.py
담당: A·B·D (공동)

핵심 분석 — RQ4(담론 vs 현실 간극) + RQ5(국내 vs 글로벌 격차)
출력: outputs/figures/rq{4,5}_*.png, outputs/rq{4,5}_*.csv

Validity Guardrail:
  #1 공정 비교 — 동일 시점(2026 뉴스 vs 2026 국내공고)
  #2 비교 어휘 분리 — RQ4a 기술/역량 사전 토큰만 / 담론 프레임(RQ4b)은 별도(08·time_series)
  #3 한국어↔영어 토픽 직접 비교 금지 — 기술스택 사전 토큰 빈도만 비교
  #4 시점 차이 명시 — RQ5는 한국 2026-05 vs 글로벌 2024-04 단면 (~24개월 차이)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.visualization._fonts import setup_korean_font
setup_korean_font()

from src.analysis.gap_index import skill_frequency, gap_table
from src.preprocessing.tech_dictionary import AI_TIERS, get_ai_tier
from src.preprocessing.role_classifier import ROLE_LABELS

KR_PATH       = "data/processed/domestic/kr_jobs_clean.csv"
NEWS_PATH     = "data/processed/news/news_processed.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

TIER_LABELS = {
    "tier_a_coding_tool":    "Tier A · AI 코딩 도구",
    "tier_b_model_platform": "Tier B · AI 모델·플랫폼",
    "tier_c_ml_skill":       "Tier C · AI/ML 직무 역량",
    "tier_d_generic":        "Tier D · 모호한 'AI'",
}
TIER_COLORS = {
    "tier_a_coding_tool":    "#d62728",
    "tier_b_model_platform": "#ff7f0e",
    "tier_c_ml_skill":       "#2ca02c",
    "tier_d_generic":        "#7f7f7f",
}


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------
def _parse_skills(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [s.strip() for s in str(value).split("|") if s.strip()]


def load_kr() -> pd.DataFrame:
    df = pd.read_csv(KR_PATH)
    df["tokens"] = df["skills"].fillna("").apply(_parse_skills)
    return df


def load_linkedin() -> pd.DataFrame | None:
    if not os.path.exists(LINKEDIN_PATH):
        print(f"[skip] {LINKEDIN_PATH} 없음 (02 미실행)")
        return None
    df = pd.read_csv(LINKEDIN_PATH)
    df["tokens"] = df["skills"].fillna("").apply(_parse_skills)
    return df


def share_per_token(df: pd.DataFrame, denom: str = "all") -> pd.Series:
    """토큰별 '해당 토큰 보유 공고 비율(%)' — 공고 단위 보유율.

    denom='all'     : 전체 공고 분모 (기본, 기존 결과 호환)
    denom='nonzero' : 스킬 1개 이상 추출된 공고만 분모 (분모 민감도 검증용)
                      → LinkedIn 추출 성공 78.7% vs 국내 97% 차이 통제
    """
    if denom == "nonzero":
        sub = df[df["tokens"].apply(lambda x: len(x) > 0)]
    else:
        sub = df
    n = len(sub)
    if n == 0:
        return pd.Series(dtype=float)
    counts = {}
    for toks in sub["tokens"]:
        for t in set(toks):
            counts[t] = counts.get(t, 0) + 1
    return (pd.Series(counts) / n * 100).sort_values(ascending=False)


def tier_share(df: pd.DataFrame, denom: str = "all") -> dict:
    """공고 단위 tier 보유율 (% · 중복 허용)."""
    if denom == "nonzero":
        sub = df[df["tokens"].apply(lambda x: len(x) > 0)]
    else:
        sub = df
    n = len(sub)
    if n == 0:
        return {t: 0.0 for t in AI_TIERS}
    out = {}
    for tier, members in AI_TIERS.items():
        members_set = set(members)
        cnt = sub["tokens"].apply(lambda toks: bool(set(toks) & members_set)).sum()
        out[tier] = cnt / n * 100
    return out


def coverage_stats(df: pd.DataFrame, label: str) -> dict:
    """추출 커버리지 통계 — 분모 민감도 분석 보조."""
    n = len(df)
    with_skills = df["tokens"].apply(lambda x: len(x) > 0).sum()
    return {"label": label, "total": n, "with_skills": with_skills,
            "coverage_pct": with_skills / n * 100 if n else 0}


# ---------------------------------------------------------------------------
# RQ5-A. 국내 vs 글로벌 사분면 (% × %)
# ---------------------------------------------------------------------------
def rq5_quadrant(kr_share: pd.Series, gl_share: pd.Series, top_n: int = 40) -> None:
    # 양쪽 보유율 상위 N 토큰 합집합
    top = sorted(set(kr_share.head(top_n).index) | set(gl_share.head(top_n).index))
    df = pd.DataFrame(index=top)
    df["kr"] = kr_share.reindex(top).fillna(0)
    df["gl"] = gl_share.reindex(top).fillna(0)
    df["gap"] = df["kr"] - df["gl"]                # +면 한국 우세, -면 글로벌 우세(lag)
    df["abs_gap"] = df["gap"].abs()
    df = df.sort_values("abs_gap", ascending=False)

    def label(row):
        kr, gl = row["kr"], row["gl"]
        if kr >= 20 and gl >= 20:
            return "공통 핵심"
        if kr - gl >= 10:
            return "한국 과대"
        if gl - kr >= 10:
            return "한국 과소(lag)"
        return "주변"
    df["quadrant"] = df.apply(label, axis=1)
    df.to_csv("outputs/rq5_gap_table.csv", encoding="utf-8-sig")

    Q_COLOR = {"공통 핵심": "#2ca02c", "한국 과대": "#d62728",
               "한국 과소(lag)": "#1f77b4", "주변": "#bbbbbb"}

    fig, ax = plt.subplots(figsize=(11, 10))
    for q, sub in df.groupby("quadrant"):
        ax.scatter(sub["gl"], sub["kr"], s=80, alpha=0.7, c=Q_COLOR[q], label=q,
                   edgecolors="white", linewidths=0.8)
    # 대각선 (한국=글로벌)
    lim = max(df["kr"].max(), df["gl"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, label="동률선")

    # 격차 큰 토큰 + 핵심 토큰 라벨
    target_show = set(df.nlargest(15, "abs_gap").index) | set(df.head(12).index)
    for sk, r in df.iterrows():
        if sk in target_show:
            ax.annotate(sk, (r["gl"], r["kr"]), fontsize=9,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("글로벌 LinkedIn 보유율 (%) → ", fontsize=11)
    ax.set_ylabel("한국 점핏·원티드 보유율 (%) →", fontsize=11)
    ax.set_title("RQ5-A: 국내 vs 글로벌 기술스택 보유율 사분면\n"
                 "한국 2026-05 활성공고 vs 글로벌 2024-04 LinkedIn 단면 (~24개월 시차)\n"
                 "※ 시점·시장(IT 전문 vs 종합 플랫폼)·본문 작성 관행 차이 결합 효과 — '관측된 격차'로 해석", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq5_quadrant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ5-A] 저장:", f"{FIG_DIR}/rq5_quadrant.png")

    print("\n=== RQ5-A 사분면 분류 ===")
    for q in ["공통 핵심", "한국 과대", "한국 과소(lag)"]:
        sub = df[df.quadrant == q].sort_values("abs_gap", ascending=False).head(10)
        print(f"\n[{q}] {len(df[df.quadrant == q])}개")
        for sk, r in sub.iterrows():
            print(f"  {sk:20s} 한국 {r.kr:5.1f}% · 글로벌 {r.gl:5.1f}% · 격차 {r.gap:+5.1f}pp")


# ---------------------------------------------------------------------------
# RQ5-B. AI 4-tier 한국 vs 글로벌 비교
# ---------------------------------------------------------------------------
def rq5_tier_compare(kr_df: pd.DataFrame, gl_df: pd.DataFrame) -> None:
    """RQ5-B: AI 4-tier 한·글 비교 — 분모 민감도 포함 (전체 분모 vs 추출성공만)."""
    tiers = list(AI_TIERS.keys())
    labels = [TIER_LABELS[t] for t in tiers]

    kr_all  = tier_share(kr_df, denom="all")
    gl_all  = tier_share(gl_df, denom="all")
    kr_nz   = tier_share(kr_df, denom="nonzero")
    gl_nz   = tier_share(gl_df, denom="nonzero")

    kr_cov = coverage_stats(kr_df, "한국")
    gl_cov = coverage_stats(gl_df, "글로벌")

    x = np.arange(len(tiers))
    w = 0.20

    fig, ax = plt.subplots(figsize=(14, 7))
    series = [
        (kr_all, "한국 (전체 분모)",     "#dd8452", -1.5*w),
        (gl_all, "글로벌 (전체 분모)",   "#4c72b0", -0.5*w),
        (kr_nz,  "한국 (스킬추출만)",   "#a4513b", +0.5*w),
        (gl_nz,  "글로벌 (스킬추출만)", "#2c4a73", +1.5*w),
    ]
    for data, label, color, off in series:
        vals = [data[t] for t in tiers]
        bars = ax.bar(x + off, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{v:.1f}", ha="center", fontsize=8)

    # 전체 분모 기준 격차 주석
    for i, t in enumerate(tiers):
        diff = kr_all[t] - gl_all[t]
        diff_nz = kr_nz[t] - gl_nz[t]
        ax.text(i, max(kr_all[t], gl_all[t], kr_nz[t], gl_nz[t]) + 5,
                f"Δ전체 {diff:+.1f}pp\nΔ추출 {diff_nz:+.1f}pp",
                ha="center", fontsize=9,
                color="darkred" if abs(diff) >= 5 else "gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("해당 tier 보유 공고 비율 (%) — 중복 허용", fontsize=11)
    ax.set_title("RQ5-B: AI 4-tier 한국 vs 글로벌 — 분모 민감도 분석\n"
                 f"한국 추출률 {kr_cov['coverage_pct']:.1f}% ({kr_cov['with_skills']}/{kr_cov['total']}) "
                 f"· 글로벌 추출률 {gl_cov['coverage_pct']:.1f}% ({gl_cov['with_skills']}/{gl_cov['total']})  "
                 f"· 한국 2026-05 vs 글로벌 2024-04 (~24개월 시차)",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_ylim(0, 60)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq5_ai_tier_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ5-B] 저장:", f"{FIG_DIR}/rq5_ai_tier_compare.png")
    print(f"  [커버리지] {kr_cov['label']} {kr_cov['coverage_pct']:.1f}% / "
          f"{gl_cov['label']} {gl_cov['coverage_pct']:.1f}%")
    for t in tiers:
        print(f"  {TIER_LABELS[t]:30s} | 한국 전체 {kr_all[t]:5.1f}% / 추출 {kr_nz[t]:5.1f}% "
              f"| 글로벌 전체 {gl_all[t]:5.1f}% / 추출 {gl_nz[t]:5.1f}% "
              f"| Δ전체 {kr_all[t]-gl_all[t]:+5.1f}pp / Δ추출 {kr_nz[t]-gl_nz[t]:+5.1f}pp")


# ---------------------------------------------------------------------------
# RQ5-C. 직무 × Tier A 한국 vs 글로벌
# ---------------------------------------------------------------------------
def rq5_role_tier_compare(kr_df: pd.DataFrame, gl_df: pd.DataFrame, tier_key: str = "tier_a_coding_tool") -> None:
    members = set(AI_TIERS[tier_key])
    common_roles = [r for r in ROLE_LABELS if r not in ("game", "other")]

    rows = []
    for role in common_roles:
        kr_sub = kr_df[kr_df["role"] == role]
        gl_sub = gl_df[gl_df["role"] == role]
        kr_rate = kr_sub["tokens"].apply(lambda t: bool(set(t) & members)).mean() * 100 if len(kr_sub) else 0
        gl_rate = gl_sub["tokens"].apply(lambda t: bool(set(t) & members)).mean() * 100 if len(gl_sub) else 0
        rows.append({"role": role, "label": ROLE_LABELS[role],
                     "kr_n": len(kr_sub), "gl_n": len(gl_sub),
                     "kr_rate": kr_rate, "gl_rate": gl_rate,
                     "gap": kr_rate - gl_rate})
    cmp = pd.DataFrame(rows).sort_values("gap", ascending=False)
    cmp.to_csv(f"outputs/rq5_role_x_{tier_key}.csv", encoding="utf-8-sig", index=False)

    labels = [f"{r['label']}\n(KR n={r['kr_n']}/GL n={r['gl_n']})" for _, r in cmp.iterrows()]
    x = np.arange(len(cmp))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - w/2, cmp["kr_rate"], w, label="한국", color="#dd8452")
    ax.bar(x + w/2, cmp["gl_rate"], w, label="글로벌", color="#4c72b0")
    for i, (_, r) in enumerate(cmp.iterrows()):
        ax.text(i - w/2, r["kr_rate"] + 0.5, f"{r['kr_rate']:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, r["gl_rate"] + 0.5, f"{r['gl_rate']:.1f}%", ha="center", fontsize=9)
        ax.text(i, max(r["kr_rate"], r["gl_rate"]) + 3,
                f"Δ {r['gap']:+.1f}", ha="center", fontsize=9,
                color="darkred" if abs(r["gap"]) >= 5 else "gray", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel(f"{TIER_LABELS[tier_key]} 보유 공고 비율 (%)", fontsize=11)
    ax.set_title(f"RQ5-C: 직무 × {TIER_LABELS[tier_key]} — 한국 vs 글로벌\n"
                 "한국 2026-05 활성공고 vs 글로벌 2024-04 LinkedIn 단면 (~24개월 시차)\n"
                 "※ 격차는 한국 진보가 아닌 시점·시장·관행의 결합 효과 — '관측된 격차'", fontsize=11)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(0, max(cmp[["kr_rate", "gl_rate"]].values.max() * 1.3, 5))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq5_role_x_{tier_key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ5-C] 저장: {FIG_DIR}/rq5_role_x_{tier_key}.png")

    print(f"\n=== RQ5-C 직무×{TIER_LABELS[tier_key]} (한국-글로벌 격차 순) ===")
    for _, r in cmp.iterrows():
        print(f"  {r['label']:12s}  한국 {r['kr_rate']:5.1f}% · 글로벌 {r['gl_rate']:5.1f}% · Δ {r['gap']:+5.1f}pp")


# ---------------------------------------------------------------------------
# RQ5-D. 핵심 토큰 격차 막대 (AI 관련 + 주요 스택)
# ---------------------------------------------------------------------------
def rq5_key_tokens_gap(kr_share: pd.Series, gl_share: pd.Series) -> None:
    HIGHLIGHT = [
        # Tier A 코딩 도구
        "Cursor", "Claude Code", "Copilot", "GitHub Copilot", "Codex",
        # Tier B 모델·플랫폼
        "ChatGPT", "Claude", "OpenAI", "GPT", "Gemini",
        # Tier C ML 역량
        "LLM", "RAG", "MLOps", "Fine-tuning", "PyTorch", "TensorFlow",
        # Tier D
        "AI",
        # 공통 핵심
        "Python", "AWS", "Docker", "Kubernetes", "React",
    ]
    rows = []
    for tok in HIGHLIGHT:
        kr = kr_share.get(tok, 0)
        gl = gl_share.get(tok, 0)
        tier = get_ai_tier(tok)
        rows.append({"token": tok, "kr": kr, "gl": gl, "gap": kr - gl,
                     "tier": tier or "non_ai"})
    df = pd.DataFrame(rows).sort_values("gap", ascending=True)

    tier_color = {
        "tier_a_coding_tool":    TIER_COLORS["tier_a_coding_tool"],
        "tier_b_model_platform": TIER_COLORS["tier_b_model_platform"],
        "tier_c_ml_skill":       TIER_COLORS["tier_c_ml_skill"],
        "tier_d_generic":        TIER_COLORS["tier_d_generic"],
        "non_ai":                "#888888",
    }
    colors = [tier_color[t] for t in df["tier"]]

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(df["token"], df["gap"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, (_, r) in zip(bars, df.iterrows()):
        sign = "+" if r["gap"] >= 0 else ""
        ax.text(r["gap"] + (1 if r["gap"] >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f"{sign}{r['gap']:.1f}pp  (KR {r['kr']:.1f} · GL {r['gl']:.1f})",
                va="center", ha="left" if r["gap"] >= 0 else "right", fontsize=9)
    ax.set_title("RQ5-D: 핵심 토큰의 국내·글로벌 관측 격차 (한국 - 글로벌, pp)\n"
                 "한국 2026-05 vs 글로벌 2024-04 (~24개월 시차) — 양/음수는 채용 요건 명시 빈도 차이일 뿐,\n"
                 "도구·역량 실 도입률은 별도 검증 필요 (Claude Code 2024-06 출시는 글로벌 데이터 이후)", fontsize=11)
    ax.set_xlabel("격차 (pp · 양수 = 한국 우세)")

    import matplotlib.patches as mpatches
    legend_items = [
        ("Tier A · AI 코딩 도구",    TIER_COLORS["tier_a_coding_tool"]),
        ("Tier B · AI 모델·플랫폼", TIER_COLORS["tier_b_model_platform"]),
        ("Tier C · AI/ML 직무 역량", TIER_COLORS["tier_c_ml_skill"]),
        ("Tier D · 모호한 'AI'",     TIER_COLORS["tier_d_generic"]),
        ("기타 핵심 스택",            "#888888"),
    ]
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in legend_items],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq5_key_tokens_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ5-D] 저장:", f"{FIG_DIR}/rq5_key_tokens_gap.png")


# ---------------------------------------------------------------------------
# (RQ4은 뉴스 데이터 도착 후 작동)
# ---------------------------------------------------------------------------
def rq4_news_vs_jobs(kr_df: pd.DataFrame) -> None:
    if not os.path.exists(NEWS_PATH):
        print(f"[RQ4] 스킵 — {NEWS_PATH} 없음 (01 미실행)")
        return
    # TODO: 뉴스 수집 후 활성화
    print("[RQ4] 뉴스 데이터 도착 시 활성화")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kr_df = load_kr()
    print(f"[국내] {len(kr_df):,}건 로드")
    kr_share = share_per_token(kr_df)
    kr_tier  = tier_share(kr_df)

    gl_df = load_linkedin()
    if gl_df is None:
        print("LinkedIn 미처리 — 02 실행 후 다시 시도")
    else:
        print(f"[글로벌] {len(gl_df):,}건 로드")
        gl_share = share_per_token(gl_df)
        gl_tier  = tier_share(gl_df)

        rq5_quadrant(kr_share, gl_share, top_n=40)
        rq5_tier_compare(kr_df, gl_df)
        rq5_role_tier_compare(kr_df, gl_df, "tier_a_coding_tool")
        rq5_role_tier_compare(kr_df, gl_df, "tier_c_ml_skill")
        rq5_key_tokens_gap(kr_share, gl_share)

    rq4_news_vs_jobs(kr_df)
