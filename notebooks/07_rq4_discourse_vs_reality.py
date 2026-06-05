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
from src.preprocessing.role_classifier import ROLE_LABELS, match_all_roles
from scipy.stats import spearmanr

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
# RQ4. 담론(뉴스) vs 현실(국내공고) 간극
# ---------------------------------------------------------------------------
# 뉴스 전처리 CSV 로더 — #7 스키마(date·skills·tier_*) 우선, 구 스키마(발행일·tech_tokens) 호환.
def _split_news_tokens(value) -> list[str]:
    """뉴스 토큰 문자열 → list. 구분자 ';'(#7) / '|'(구 스키마) / 공백 자동 인식."""
    if pd.isna(value):
        return []
    s = str(value)
    for sep in (";", "|"):
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [t for t in s.split() if t] if s else []


def load_news_processed() -> pd.DataFrame | None:
    """data/processed/news/news_processed.csv 로드 + year·news_tokens 정규화.

    Guardrail #1(공정 비교): year 컬럼으로 2026만 골라 공고(2026 스냅샷)와 동일 시점 비교.
    Guardrail #3: news_tokens 는 기술스택 사전(extract_skills) canonical 토큰 — 공고와 동일 단위.
    """
    if not os.path.exists(NEWS_PATH):
        print(f"[RQ4] 스킵 — {NEWS_PATH} 없음 (01 미실행)")
        return None
    news = pd.read_csv(NEWS_PATH)

    # 연도 컬럼 확보 (date / 발행일 어느 쪽이든)
    if "year" not in news.columns:
        date_col = next((c for c in ("date", "발행일") if c in news.columns), None)
        if date_col is None:
            raise KeyError("뉴스 CSV에 year·date·발행일 컬럼이 모두 없습니다")
        news["year"] = pd.to_datetime(news[date_col], errors="coerce").dt.year

    # 기술토큰 컬럼 확보 (skills[#7] / tech_tokens[구])
    tok_col = next((c for c in ("skills", "tech_tokens") if c in news.columns), None)
    if tok_col is None:
        raise KeyError("뉴스 CSV에 skills·tech_tokens 컬럼이 모두 없습니다")
    news["news_tokens"] = news[tok_col].apply(_split_news_tokens)
    return news


Q4_COLOR = {
    "담론 과잉(뉴스>공고)":   "#d62728",
    "조용한 핵심(공고>뉴스)": "#1f77b4",
    "공통":                   "#2ca02c",
}


def rq4_news_vs_jobs(kr_df: pd.DataFrame) -> None:
    """RQ4a. 뉴스 담론 vs 실제 공고의 기술토큰 순위 비교 — Spearman + 사분면.

    우하단=담론만 뜨거움(뉴스>공고), 좌상단=조용한 핵심(공고는 쓰는데 담론은 조용).
    """
    news = load_news_processed()
    if news is None:
        return
    news_2026 = news[news["year"] == 2026]
    if len(news_2026) == 0:
        print("[RQ4a] 스킵 — 2026년 뉴스 0건")
        return
    print(f"[RQ4a] 뉴스 {len(news):,}건 중 2026년 {len(news_2026):,}건 사용 (공고와 동일 시점)")

    news_freq = skill_frequency(news_2026["news_tokens"])
    job_freq  = skill_frequency(kr_df["tokens"])

    df, rho, p = gap_table(news_freq, job_freq, top_n=40)
    df.to_csv("outputs/rq4_gap_table.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 10))
    for q, sub in df.groupby("quadrant"):
        ax.scatter(sub["news_rank"], sub["job_rank"], s=90, alpha=0.75,
                   c=Q4_COLOR.get(q, "#999999"), label=q, edgecolors="white", linewidths=0.8)
    lim = max(df["news_rank"].max(), df["job_rank"].max()) + 2
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, label="순위 일치선")
    for sk, r in df.iterrows():
        ax.annotate(sk, (r["news_rank"], r["job_rank"]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("← 뉴스 담론 순위 (1=가장 많이 언급)", fontsize=11)
    ax.set_ylabel("← 국내공고 순위 (1=가장 많이 요구)", fontsize=11)
    ax.set_title("RQ4a: 담론(뉴스) vs 현실(국내공고) 기술스택 간극\n"
                 f"2026 뉴스 {len(news_2026):,}건 × 국내공고 {len(kr_df):,}건 · "
                 f"Spearman ρ={rho:.2f} (p={p:.2g})\n"
                 "※ 우하단=담론만 뜨거움, 좌상단=조용한 핵심(현장은 쓰는데 담론은 조용)", fontsize=11)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq4_gap_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ4a] 저장: {FIG_DIR}/rq4_gap_index.png · Spearman ρ={rho:.3f} (p={p:.3g})")
    for q in ["담론 과잉(뉴스>공고)", "조용한 핵심(공고>뉴스)"]:
        toks = list(df[df.quadrant == q].index[:8])
        print(f"  [{q}] {toks}")


def rq4_tier_gap(kr_df: pd.DataFrame) -> None:
    """RQ4b. AI 4-tier 담론(뉴스 2026) vs 현실(국내공고 2026) 보유율 격차.

    "뉴스는 Tier B(모델·플랫폼)를 X% 외치는데 실제 공고엔 Y%만 명시되는가?"
    #7 전처리는 tier_a~d 컬럼을 직접 제공 → 있으면 그대로, 없으면 skills로 계산.
    뉴스=본문 멀티라벨 언급 / 공고=요구역량 명시 — 단위가 비대칭이므로 '관측된 간극'으로 해석.
    """
    news = load_news_processed()
    if news is None:
        return
    news_2026 = news[news["year"] == 2026]
    if len(news_2026) == 0:
        print("[RQ4b] 스킵 — 2026년 뉴스 0건")
        return

    tiers = list(AI_TIERS.keys())
    # 뉴스 담론 tier 보유율 — tier_* 컬럼 직접(#7) 또는 토큰 계산(구 스키마)
    if set(tiers).issubset(news_2026.columns):
        news_tier = {t: news_2026[t].mean() * 100 for t in tiers}
        src_note = "tier 컬럼 직접"
    else:
        news_tier = {}
        for t in tiers:
            members = set(AI_TIERS[t])
            news_tier[t] = news_2026["news_tokens"].apply(
                lambda toks: bool(set(toks) & members)).mean() * 100
        src_note = "skills 토큰 계산"
    job_tier = tier_share(kr_df)  # 공고 단위 보유율 dict (기존 함수 재사용)

    rows = [{"tier": t, "label": TIER_LABELS[t],
             "news": news_tier[t], "job": job_tier[t],
             "gap": news_tier[t] - job_tier[t]} for t in tiers]
    cmp = pd.DataFrame(rows)
    cmp.to_csv("outputs/rq4_tier_gap.csv", encoding="utf-8-sig", index=False)

    x = np.arange(len(tiers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - w/2, cmp["news"], w, label="뉴스 담론 (2026)", color="#d62728")
    ax.bar(x + w/2, cmp["job"],  w, label="국내공고 현실 (2026)", color="#1f77b4")
    for i, r in cmp.iterrows():
        ax.text(i - w/2, r["news"] + 0.5, f"{r['news']:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, r["job"] + 0.5, f"{r['job']:.1f}%", ha="center", fontsize=9)
        ax.text(i, max(r["news"], r["job"]) + 3, f"Δ {r['gap']:+.1f}pp",
                ha="center", fontsize=10, fontweight="bold",
                color="darkred" if abs(r["gap"]) >= 5 else "gray")
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in tiers], fontsize=10)
    ax.set_ylabel("해당 tier 보유 비율 (%) — 중복 허용", fontsize=11)
    ax.set_title("RQ4b: AI 4-tier 담론(뉴스) vs 현실(공고) 간극 — 동일 2026 시점\n"
                 "양수 Δ = 담론 과잉(뉴스가 더 많이 호명) · 음수 Δ = 현장이 더 요구\n"
                 "※ 뉴스=본문 멀티라벨 언급 / 공고=요구역량 명시 — '관측된 간극'으로 해석", fontsize=11)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq4_tier_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ4b] 저장: {FIG_DIR}/rq4_tier_gap.png (뉴스 tier 출처: {src_note})")
    for r in rows:
        print(f"  {r['label']:24s} 뉴스 {r['news']:5.1f}% · 공고 {r['job']:5.1f}% · Δ {r['gap']:+5.1f}pp")


# ---------------------------------------------------------------------------
# RQ4-c. 담론 과잉 / 조용한 핵심 top-N 막대 (산점도 보완)
# ---------------------------------------------------------------------------
def rq4_top_gap_bars(kr_df: pd.DataFrame, top_k: int = 15) -> None:
    """뉴스 담론과 공고 현실의 순위 격차를 막대로 — 산점도가 한눈에 안 들어와서 보완."""
    news = load_news_processed()
    if news is None:
        return
    news_2026 = news[news["year"] == 2026]
    if len(news_2026) == 0:
        print("[RQ4c] 스킵 — 2026년 뉴스 0건")
        return

    news_freq = skill_frequency(news_2026["news_tokens"])
    job_freq  = skill_frequency(kr_df["tokens"])
    # 양쪽 모두에 등장하는 토큰만 — 한쪽 0인 noise 컷
    common = [t for t in news_freq.index if t in job_freq.index]
    df = pd.DataFrame(index=common)
    df["news_freq"] = news_freq.reindex(common)
    df["job_freq"]  = job_freq.reindex(common)
    df["news_rank"] = df["news_freq"].rank(ascending=False, method="min")
    df["job_rank"]  = df["job_freq"].rank(ascending=False, method="min")
    df["delta_rank"] = df["news_rank"] - df["job_rank"]  # +=조용한핵심, -=담론과잉
    # 너무 마이너 토큰 제외 — 양쪽 top 100 안에 들어야 의미 있음
    df = df[(df["news_rank"] <= 100) | (df["job_rank"] <= 100)]
    df.to_csv("outputs/rq4_top_gap.csv", encoding="utf-8-sig")

    discourse_only = df.sort_values("delta_rank", ascending=True).head(top_k)  # 가장 음수 → 담론 과잉
    quiet_core     = df.sort_values("delta_rank", ascending=False).head(top_k)  # 가장 양수 → 조용한 핵심

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # 좌: 담론 과잉 (뉴스>공고)
    ax = axes[0]
    y = np.arange(len(discourse_only))
    ax.barh(y, -discourse_only["delta_rank"], color="#d62728", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(discourse_only.index, fontsize=10)
    ax.invert_yaxis()
    for i, (sk, r) in enumerate(discourse_only.iterrows()):
        ax.text(-r["delta_rank"] + 1, i,
                f"뉴스 #{int(r.news_rank)} → 공고 #{int(r.job_rank)}",
                va="center", fontsize=8, color="#444")
    ax.set_xlabel("순위 격차 (뉴스가 더 상위 = 담론 과잉)", fontsize=10)
    ax.set_title(f"담론 과잉 Top {top_k}\n뉴스 담론은 강하지만 공고 요구는 약함", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    # 우: 조용한 핵심 (공고>뉴스)
    ax = axes[1]
    y = np.arange(len(quiet_core))
    ax.barh(y, quiet_core["delta_rank"], color="#1f77b4", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(quiet_core.index, fontsize=10)
    ax.invert_yaxis()
    for i, (sk, r) in enumerate(quiet_core.iterrows()):
        ax.text(r["delta_rank"] + 1, i,
                f"뉴스 #{int(r.news_rank)} → 공고 #{int(r.job_rank)}",
                va="center", fontsize=8, color="#444")
    ax.set_xlabel("순위 격차 (공고가 더 상위 = 조용한 핵심)", fontsize=10)
    ax.set_title(f"조용한 핵심 Top {top_k}\n현장은 요구하는데 담론은 조용", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("RQ4c: 담론 vs 현실 — 순위 격차 Top 토큰 (2026 뉴스 vs 공고)\n"
                 "양쪽 top 100 진입 토큰 한정 · 빨강=뉴스만 뜨거움 · 파랑=공고만 요구",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq4_top_gap_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ4c] 저장: {FIG_DIR}/rq4_top_gap_bars.png")
    print(f"  [담론 과잉 top 5] {list(discourse_only.index[:5])}")
    print(f"  [조용한 핵심 top 5] {list(quiet_core.index[:5])}")


# ---------------------------------------------------------------------------
# RQ4-d. 시점별 Spearman ρ — 담론과 현실은 시간이 갈수록 가까워졌나?
# ---------------------------------------------------------------------------
def rq4_period_spearman(kr_df: pd.DataFrame) -> None:
    """before(~2022-11) / after(2022-12~) / 2026 — 시점별 뉴스 담론 ↔ 공고 Spearman ρ.

    공고는 2026-05 단면 고정(시점 무관). 뉴스 시점만 바꿔서 '담론이 언제 현실과 가장 가까웠나' 추적.
    """
    news = load_news_processed()
    if news is None:
        return

    job_freq = skill_frequency(kr_df["tokens"])

    periods = [
        ("before",  news[news["year"] <= 2022][news["month"] <= "2022-11"] if "month" in news.columns
                                                                          else news[news["year"] <= 2022],
                    "before ChatGPT (~2022-11)"),
        ("after",   news[news["year"] >= 2023] if "year" in news.columns else news,
                    "after ChatGPT (2023~)"),
        ("y2026",   news[news["year"] == 2026], "2026 (동일 시점)"),
    ]
    # period 컬럼이 있으면 before/after는 그걸 우선 사용 (RQ3 일관성)
    if "period" in news.columns:
        periods = [
            ("before", news[news["period"] == "before"], "before ChatGPT (~2022-11)"),
            ("after",  news[news["period"] == "after"],  "after ChatGPT (2022-12~)"),
            ("y2026",  news[news["year"] == 2026],       "2026 (동일 시점)"),
        ]

    rows = []
    for key, sub, label in periods:
        if len(sub) == 0:
            print(f"[RQ4d] {label}: 0건 — 스킵")
            continue
        n_freq = skill_frequency(sub["news_tokens"])
        common_tokens = sorted(set(n_freq.index) & set(job_freq.index))
        if len(common_tokens) < 5:
            print(f"[RQ4d] {label}: 공통 토큰 부족 — 스킵")
            continue
        n_rank = n_freq.reindex(common_tokens).rank(ascending=False, method="min")
        j_rank = job_freq.reindex(common_tokens).rank(ascending=False, method="min")
        rho, p = spearmanr(n_rank, j_rank)
        rows.append({"period": key, "label": label, "n_articles": len(sub),
                     "n_common_tokens": len(common_tokens),
                     "rho": rho, "p_value": p})
        print(f"[RQ4d] {label}: n={len(sub):,} · 공통토큰 {len(common_tokens)} · ρ={rho:.3f} (p={p:.2g})")

    if not rows:
        return
    cmp = pd.DataFrame(rows)
    cmp.to_csv("outputs/rq4_period_spearman.csv", encoding="utf-8-sig", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#aaaaaa", "#ff7f0e", "#1f77b4"][:len(cmp)]
    bars = ax.bar(cmp["label"], cmp["rho"], color=colors, edgecolor="white", width=0.55)
    for bar, r in zip(bars, cmp.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"ρ={r.rho:.3f}\nn={r.n_articles:,}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Spearman ρ — 뉴스 담론 ↔ 공고 요구역량 순위상관", fontsize=11)
    ax.set_ylim(min(0, cmp["rho"].min() - 0.1), max(cmp["rho"].max() + 0.15, 1.0))
    ax.set_title("RQ4d: 시점별 담론↔현실 순위상관 — 담론이 현실과 가까워졌나?\n"
                 "공고는 2026-05 단면 고정 / 뉴스만 시점 변화 · ρ 클수록 일치",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq4_period_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ4d] 저장: {FIG_DIR}/rq4_period_spearman.png")


# ---------------------------------------------------------------------------
# RQ4-e. 직무 × tier 격차 — 어느 직무에서 담론↔현실 간극이 가장 큰가?
# ---------------------------------------------------------------------------
def _add_news_roles(news_2026: pd.DataFrame) -> pd.DataFrame:
    """뉴스 title 기반 직무 멀티라벨 (보수적 매칭).

    body까지 보면 부수 언급 노이즈가 큼. title 매칭은 '명백히 X 직무에 관한 기사'만 잡힘.
    role_<key> 컬럼 (0/1) 추가.
    """
    if "title" not in news_2026.columns:
        return news_2026
    out = news_2026.copy()
    role_sets = out["title"].fillna("").apply(match_all_roles)
    for role in ROLE_LABELS:
        if role == "other":
            continue
        out[f"role_{role}"] = role_sets.apply(lambda s, r=role: int(r in s))
    return out


def rq4_role_x_tier_gap(kr_df: pd.DataFrame) -> None:
    """직무 × tier 격차 히트맵 — 뉴스 담론 보유율 - 공고 현실 보유율 (pp)."""
    news = load_news_processed()
    if news is None:
        return
    news_2026 = news[news["year"] == 2026]
    if len(news_2026) == 0:
        print("[RQ4e] 스킵 — 2026년 뉴스 0건")
        return

    news_2026 = _add_news_roles(news_2026)
    tiers = list(AI_TIERS.keys())
    target_roles = [r for r in ROLE_LABELS if r not in ("other", "game")]

    rows = []
    for role in target_roles:
        role_col = f"role_{role}"
        if role_col not in news_2026.columns:
            continue
        news_sub = news_2026[news_2026[role_col] == 1]
        kr_sub   = kr_df[kr_df["role"] == role]
        if len(news_sub) < 10 or len(kr_sub) < 10:
            # 표본 너무 작으면 스킵 (오차 큼)
            continue
        for tier in tiers:
            news_rate = news_sub[tier].mean() * 100 if tier in news_sub.columns else float("nan")
            members = set(AI_TIERS[tier])
            job_rate = kr_sub["tokens"].apply(lambda toks: bool(set(toks) & members)).mean() * 100
            rows.append({
                "role": role, "role_label": ROLE_LABELS[role],
                "tier": tier, "tier_label": TIER_LABELS[tier],
                "news_n": len(news_sub), "job_n": len(kr_sub),
                "news_rate": news_rate, "job_rate": job_rate,
                "gap": news_rate - job_rate,
            })
    if not rows:
        print("[RQ4e] 스킵 — 직무×tier 매칭 표본 부족")
        return
    df = pd.DataFrame(rows)
    df.to_csv("outputs/rq4_role_x_tier_gap.csv", encoding="utf-8-sig", index=False)

    # 히트맵용 pivot — gap (pp)
    pivot = df.pivot(index="role_label", columns="tier_label", values="gap")
    # 직무 정렬 — RQ3 색상 팔레트 순서와 맞추기 위해 ai_ml 위로
    role_order_by_key = ["ai_ml", "data", "devops", "security", "embedded", "qa",
                         "backend", "frontend", "fullstack", "mobile"]
    ordered = [ROLE_LABELS[r] for r in role_order_by_key if ROLE_LABELS[r] in pivot.index]
    pivot = pivot.reindex(ordered)
    tier_order = [TIER_LABELS[t] for t in tiers]
    pivot = pivot[tier_order]

    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = float(pivot.abs().max().max()) or 1.0
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(tier_order)))
    ax.set_xticklabels(tier_order, fontsize=10, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=11)
    # 표본 크기 라벨 우측 부착
    sample_n = {ROLE_LABELS[r]: (df[df.role == r]["news_n"].iloc[0],
                                  df[df.role == r]["job_n"].iloc[0])
                for r in role_order_by_key
                if r in df["role"].unique()}
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(np.arange(len(ordered)))
    ax2.set_yticklabels([f"뉴스 n={sample_n[r][0]} / 공고 n={sample_n[r][1]}"
                         for r in ordered], fontsize=8, color="#666")
    for i, r_label in enumerate(ordered):
        for j, t_label in enumerate(tier_order):
            v = pivot.iloc[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if abs(v) > vmax * 0.55 else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.10)
    cb.set_label("담론 - 현실 (pp) · +=뉴스 과잉 · -=공고 과잉", fontsize=10)
    ax.set_title("RQ4e: 직무 × AI tier 담론↔현실 간극 (2026 뉴스 vs 공고)\n"
                 "셀 값 = 뉴스 보유율% - 공고 보유율% (pp) · "
                 "빨강=뉴스가 더 호명 / 파랑=공고가 더 요구\n"
                 "※ 뉴스 직무 매칭은 title 기반 보수적 멀티라벨 · n<10 직무는 제외",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rq4_role_x_tier_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ4e] 저장: {FIG_DIR}/rq4_role_x_tier_gap.png")
    for role in role_order_by_key:
        sub = df[df["role"] == role]
        if sub.empty:
            continue
        gaps = " · ".join(f"{TIER_LABELS[r.tier].split(' · ')[0]} Δ{r.gap:+.1f}"
                          for r in sub.itertuples())
        print(f"  [{ROLE_LABELS[role]:8s}] 뉴스 n={sub['news_n'].iloc[0]} / 공고 n={sub['job_n'].iloc[0]} | {gaps}")


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
    rq4_tier_gap(kr_df)
    rq4_top_gap_bars(kr_df)
    rq4_period_spearman(kr_df)
    rq4_role_x_tier_gap(kr_df)
