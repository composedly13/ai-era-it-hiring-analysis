"""
04_eda_keyword_analysis.py
담당: 전원

탐색적 분석 — 빈도·AI 4-tier 분해·WordCloud·Co-occurrence·시계열
RQ1(국내공고 상위 역량 + AI 4-tier 분해) / RQ2(LinkedIn 글로벌) / RQ3(뉴스 담론)
출력: outputs/figures/
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ast
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization._fonts import setup_korean_font
setup_korean_font()

from src.analysis.frequency_analysis import token_frequency
from src.analysis.time_series import (
    monthly_count, monthly_skill_rate, monthly_tier_share, yearly_tier_share,
    period_token_freq, monthly_frame_intensity, FRAMING_KEYWORDS, CHATGPT_LAUNCH,
)
from src.analysis.cooccurrence_network import build_cooccurrence_matrix, build_graph
from src.visualization.wordcloud_gen import generate_wordcloud
from src.visualization.network_plot import draw_network
from src.preprocessing.tech_dictionary import AI_TIERS, TECH_DICT, get_ai_tier
from src.preprocessing.role_classifier import ROLE_LABELS, match_all_roles

KR_PATH       = "data/processed/domestic/kr_jobs_clean.csv"
NEWS_PATH     = "data/processed/news/news_processed.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"

os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 카테고리·색상 매핑
# ---------------------------------------------------------------------------
TIER_LABELS = {
    "tier_a_coding_tool":   "Tier A · AI 코딩 도구",
    "tier_b_model_platform":"Tier B · AI 모델·플랫폼",
    "tier_c_ml_skill":      "Tier C · AI/ML 직무 역량",
    "tier_d_generic":       "Tier D · AI 일반어(광의)",
}
TIER_COLORS = {
    "tier_a_coding_tool":   "#d62728",
    "tier_b_model_platform":"#ff7f0e",
    "tier_c_ml_skill":      "#2ca02c",
    "tier_d_generic":       "#7f7f7f",
}
CAT_COLORS = {
    "language":          "#1f77b4",
    "frontend":          "#17becf",
    "backend":           "#9467bd",
    "mobile":            "#bcbd22",
    "database":          "#8c564b",
    "cloud_infra":       "#e377c2",
    "data_eng":          "#7f7f7f",
    "collaboration":     "#aec7e8",
    "ai_coding_tool":    TIER_COLORS["tier_a_coding_tool"],
    "ai_model_platform": TIER_COLORS["tier_b_model_platform"],
    "ai_ml_skill":       TIER_COLORS["tier_c_ml_skill"],
    "ai_generic":        TIER_COLORS["tier_d_generic"],
    "기타":              "#cccccc",
}

# canonical → category
_SKILL2CAT = {s: cat for cat, skills in TECH_DICT.items() for s in skills}


def category_of(skill: str) -> str:
    return _SKILL2CAT.get(skill, "기타")


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------
def parse_skills(value) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, str):
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else []
            except (SyntaxError, ValueError):
                return []
        return [s.strip() for s in value.split("|") if s.strip()]
    return []


def load_token_lists(df: pd.DataFrame, col: str) -> list[list[str]]:
    return [parse_skills(v) for v in df[col]]


# ---------------------------------------------------------------------------
# RQ1-A. 국내공고 상위 기술스택 (카테고리 색상)
# ---------------------------------------------------------------------------
def rq1_kr_jobs_top_skills() -> pd.Series:
    df = pd.read_csv(KR_PATH)
    token_lists = load_token_lists(df, "skills")
    freq = token_frequency(token_lists)
    print(f"[RQ1-A] 국내공고 {len(df):,}건 · 유니크 스킬 {len(freq):,}개 · 상위 5 → {freq.head(5).to_dict()}")

    top20 = freq.head(20)
    colors = [CAT_COLORS[category_of(s)] for s in top20.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.set_title("RQ1-A: 국내 IT 채용공고 상위 20 요구 기술스택 (점핏·원티드 3,274건)", fontsize=13)
    ax.set_xlabel("언급 공고 수")
    for bar, v in zip(bars, top20.values[::-1]):
        ax.text(v + max(top20) * 0.005, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)

    legend_cats = sorted({category_of(s) for s in top20.index},
                        key=lambda c: -sum(top20.get(t, 0) for t in TECH_DICT.get(c, [])))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c]) for c in legend_cats]
    ax.legend(handles, legend_cats, loc="lower right", fontsize=9, title="카테고리")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_top_skills.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-A] 저장:", f"{FIG_DIR}/kr_top_skills.png")
    return freq


# ---------------------------------------------------------------------------
# RQ1-B. AI 4-tier 분해 — "AI 1670건" 정체 분석
# ---------------------------------------------------------------------------
def rq1_ai_tier_breakdown() -> dict:
    df = pd.read_csv(KR_PATH)
    n_total = len(df)
    job_tiers = []   # 각 공고가 보유한 tier 집합
    for s in df["skills"].fillna(""):
        tokens = set(s.split("|")) if s else set()
        tiers = {get_ai_tier(t) for t in tokens if get_ai_tier(t)}
        job_tiers.append(tiers)

    tier_counts = {t: sum(1 for js in job_tiers if t in js) for t in AI_TIERS}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [TIER_LABELS[t] for t in AI_TIERS]
    counts = [tier_counts[t] for t in AI_TIERS]
    colors = [TIER_COLORS[t] for t in AI_TIERS]

    bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1])
    ax.set_title("RQ1-B: AI 4-tier 보유율 — 어떤 의미의 AI가 등장하는가? (중복 허용 · 국내공고 3,274건)",
                 fontsize=13)
    ax.set_xlabel("해당 tier 토큰 1개 이상 포함한 공고 수 (한 공고가 여러 tier에 중복 카운트 가능)")
    for bar, c in zip(bars, counts[::-1]):
        ax.text(c + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{c}건 ({c/n_total*100:.1f}%)", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_ai_tier_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-B] 저장:", f"{FIG_DIR}/kr_ai_tier_breakdown.png")
    for t, c in tier_counts.items():
        print(f"  {TIER_LABELS[t]:30s} {c:5d}건 ({c/n_total*100:5.1f}%) [중복 허용]")
    return tier_counts


# ---------------------------------------------------------------------------
# RQ1-B-2. 상호배타 AI 사용 모드 — 우선순위 A > B > C > D-only > 미언급
# ---------------------------------------------------------------------------
def rq1_ai_exclusive_mode() -> dict:
    """한 공고 = 하나의 AI 사용 모드. 합이 정확히 총 공고 수와 일치 (배타적 분류).

    우선순위: Tier A 보유 → 'A 사용' / 아니면 B → 'B 사용' / ... / 'AI 미언급'
    """
    df = pd.read_csv(KR_PATH)
    n_total = len(df)
    members_by_tier = {t: set(AI_TIERS[t]) for t in AI_TIERS}

    mode_counts = {"Tier A · AI 코딩 도구": 0, "Tier B · AI 모델·플랫폼": 0,
                   "Tier C · AI/ML 직무 역량": 0, "Tier D · AI 일반어(광의)만": 0,
                   "AI 미언급": 0}
    mode_colors = {"Tier A · AI 코딩 도구": TIER_COLORS["tier_a_coding_tool"],
                   "Tier B · AI 모델·플랫폼": TIER_COLORS["tier_b_model_platform"],
                   "Tier C · AI/ML 직무 역량": TIER_COLORS["tier_c_ml_skill"],
                   "Tier D · AI 일반어(광의)만":   TIER_COLORS["tier_d_generic"],
                   "AI 미언급": "#e0e0e0"}

    for s in df["skills"].fillna(""):
        tokens = set(s.split("|")) if s else set()
        if tokens & members_by_tier["tier_a_coding_tool"]:
            mode_counts["Tier A · AI 코딩 도구"] += 1
        elif tokens & members_by_tier["tier_b_model_platform"]:
            mode_counts["Tier B · AI 모델·플랫폼"] += 1
        elif tokens & members_by_tier["tier_c_ml_skill"]:
            mode_counts["Tier C · AI/ML 직무 역량"] += 1
        elif tokens & members_by_tier["tier_d_generic"]:
            mode_counts["Tier D · AI 일반어(광의)만"] += 1
        else:
            mode_counts["AI 미언급"] += 1

    assert sum(mode_counts.values()) == n_total, "상호배타 분류 합계 불일치"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = list(mode_counts.keys())
    counts = list(mode_counts.values())
    colors = [mode_colors[l] for l in labels]
    bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1])
    ax.set_title(f"RQ1-B-2: AI 사용 모드 (상호배타 분류, 우선순위 A>B>C>D-only>미언급 · {n_total:,}건)",
                 fontsize=13)
    ax.set_xlabel("공고 수 (합계 = 총 공고 수, 한 공고 = 하나의 모드)")
    for bar, c in zip(bars, counts[::-1]):
        ax.text(c + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{c}건 ({c/n_total*100:.1f}%)", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_ai_exclusive_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-B-2] 저장:", f"{FIG_DIR}/kr_ai_exclusive_mode.png  [상호배타 · 합계 검증 통과]")
    for m, c in mode_counts.items():
        print(f"  {m:30s} {c:5d}건 ({c/n_total*100:5.1f}%)")
    return mode_counts


# ---------------------------------------------------------------------------
# RQ1-C. Tier A·Tier C 상세 — 어떤 도구·어떤 ML 역량인가
# ---------------------------------------------------------------------------
def rq1_ai_tier_detail() -> None:
    df = pd.read_csv(KR_PATH)
    token_lists = load_token_lists(df, "skills")
    freq = token_frequency(token_lists)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, tier_key in zip(axes, ["tier_a_coding_tool", "tier_c_ml_skill"]):
        members = AI_TIERS[tier_key]
        member_freq = pd.Series({m: freq.get(m, 0) for m in members}).sort_values()
        ax.barh(member_freq.index, member_freq.values, color=TIER_COLORS[tier_key])
        ax.set_title(TIER_LABELS[tier_key], fontsize=12)
        ax.set_xlabel("언급 공고 수")
        for i, v in enumerate(member_freq.values):
            ax.text(v + max(member_freq.values) * 0.01, i, str(v), va="center", fontsize=9)

    fig.suptitle("RQ1-C: AI 코딩 도구(Tier A) vs AI/ML 직무 역량(Tier C) — 상세 빈도",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_ai_tier_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-C] 저장:", f"{FIG_DIR}/kr_ai_tier_detail.png")


# ---------------------------------------------------------------------------
# RQ1-D. "AI 1670" 중복 패턴 — Tier D 단독 vs 다른 tier 동시 보유
# ---------------------------------------------------------------------------
def rq1_ai_overlap() -> None:
    df = pd.read_csv(KR_PATH)
    breakdown = {"AI only (D 단독)": 0, "AI + Tier A": 0, "AI + Tier B": 0,
                 "AI + Tier C": 0, "AI + 복수 tier": 0}
    for s in df["skills"].fillna(""):
        tokens = set(s.split("|")) if s else set()
        if "AI" not in tokens:
            continue
        other_tiers = {get_ai_tier(t) for t in tokens if get_ai_tier(t) and get_ai_tier(t) != "tier_d_generic"}
        if not other_tiers:
            breakdown["AI only (D 단독)"] += 1
        elif len(other_tiers) > 1:
            breakdown["AI + 복수 tier"] += 1
        elif "tier_a_coding_tool" in other_tiers:
            breakdown["AI + Tier A"] += 1
        elif "tier_b_model_platform" in other_tiers:
            breakdown["AI + Tier B"] += 1
        elif "tier_c_ml_skill" in other_tiers:
            breakdown["AI + Tier C"] += 1

    total = sum(breakdown.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    colors_ = ["#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    bars = ax.barh(labels[::-1], values[::-1], color=colors_[::-1])
    ax.set_title(f"RQ1-D: 'AI' 언급 {total}건의 동반 tier 분포 — 추상명사인가 구체 활용인가?",
                 fontsize=12)
    ax.set_xlabel("공고 수")
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + max(values) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v}건 ({v/total*100:.1f}%)", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_ai_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-D] 저장:", f"{FIG_DIR}/kr_ai_overlap.png")
    for k, v in breakdown.items():
        print(f"  {k:25s} {v:5d}건 ({v/total*100:5.1f}%)")


# ---------------------------------------------------------------------------
# RQ1-E. Co-occurrence Network (카테고리 색상)
# ---------------------------------------------------------------------------
def rq1_kr_jobs_cooccurrence(min_count: int = 20, top_n: int = 70) -> None:
    df = pd.read_csv(KR_PATH)
    token_lists = load_token_lists(df, "skills")
    edges = build_cooccurrence_matrix(token_lists, min_count=min_count)
    G = build_graph(edges)
    print(f"[RQ1-E·Co-occ] 노드 {G.number_of_nodes()} · 엣지 {G.number_of_edges()} (min_count={min_count})")

    node_color_fn = lambda n: CAT_COLORS[category_of(n)]
    draw_network(G, title="RQ1-E: 국내공고 기술스택 동시출현 네트워크 (카테고리 색상)",
                 save_path=f"{FIG_DIR}/cooccurrence_network.png",
                 top_n=top_n, node_color_fn=node_color_fn,
                 legend={cat: CAT_COLORS[cat] for cat in
                         ["language", "frontend", "backend", "database", "cloud_infra",
                          "ai_coding_tool", "ai_model_platform", "ai_ml_skill", "ai_generic",
                          "collaboration", "data_eng"]})
    print("[RQ1-E] 저장:", f"{FIG_DIR}/cooccurrence_network.png")


def rq1_wordcloud(freq: pd.Series) -> None:
    generate_wordcloud(freq.to_dict(), title="국내공고 기술스택", save_path=f"{FIG_DIR}/wc_kr_jobs.png")
    print("[RQ1·WC] 저장:", f"{FIG_DIR}/wc_kr_jobs.png")


# ---------------------------------------------------------------------------
# RQ1-F. 직무군 분포
# ---------------------------------------------------------------------------
ROLE_ORDER = ["ai_ml", "backend", "devops", "embedded", "frontend",
              "security", "data", "fullstack", "qa", "mobile", "game", "other"]
ROLE_COLORS = {
    "ai_ml":     "#2ca02c",
    "backend":   "#9467bd",
    "devops":    "#e377c2",
    "embedded":  "#8c564b",
    "frontend":  "#17becf",
    "security":  "#bcbd22",
    "data":      "#7f7f7f",
    "fullstack": "#ff7f0e",
    "qa":        "#1f77b4",
    "mobile":    "#aec7e8",
    "game":      "#d62728",
    "other":     "#cccccc",
}


def rq1_role_distribution() -> pd.Series:
    df = pd.read_csv(KR_PATH)
    dist = df["role"].value_counts().reindex(ROLE_ORDER).fillna(0).astype(int)
    n_total = int(dist.sum())

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [f"{ROLE_LABELS[r]}" for r in dist.index]
    colors = [ROLE_COLORS[r] for r in dist.index]
    bars = ax.barh(labels[::-1], dist.values[::-1], color=colors[::-1])
    ax.set_title(f"RQ1-F: 국내공고 직무군 분포 ({n_total:,}건)", fontsize=13)
    ax.set_xlabel("공고 수")
    for bar, v in zip(bars, dist.values[::-1]):
        ax.text(v + dist.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v}건 ({v/n_total*100:.1f}%)", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_role_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-F] 저장:", f"{FIG_DIR}/kr_role_distribution.png")
    print(f"  분류율: {(n_total - dist.get('other', 0))/n_total*100:.1f}% ({n_total - dist.get('other', 0)}/{n_total})")
    return dist


# ---------------------------------------------------------------------------
# RQ1-G. 직무군별 상위 스킬 (small multiples)
# ---------------------------------------------------------------------------
def rq1_top_skills_by_role(top_n: int = 7) -> None:
    df = pd.read_csv(KR_PATH)
    # other 제외, game은 표본 적어 제외 (8건)
    roles = [r for r in ROLE_ORDER if r not in ("other", "game")]

    n_cols = 3
    n_rows = (len(roles) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows))
    axes = axes.flatten()

    for ax, role in zip(axes, roles):
        sub = df[df["role"] == role]
        token_lists = load_token_lists(sub, "skills")
        freq = token_frequency(token_lists).head(top_n)
        ax.barh(freq.index[::-1], freq.values[::-1], color=ROLE_COLORS[role])
        ax.set_title(f"{ROLE_LABELS[role]}  ({len(sub)}건)", fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        for i, v in enumerate(freq.values[::-1]):
            ax.text(v + freq.max() * 0.01, i, str(v), va="center", fontsize=8)

    for ax in axes[len(roles):]:
        ax.set_visible(False)

    fig.suptitle("RQ1-G: 직무군별 상위 7 기술스택", fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_top_skills_by_role.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-G] 저장:", f"{FIG_DIR}/kr_top_skills_by_role.png")


# ---------------------------------------------------------------------------
# RQ1-H. 직무군 × AI 4-tier 히트맵 — 어떤 직무가 어떤 AI를 요구하나
# ---------------------------------------------------------------------------
def rq1_role_x_ai_tier() -> None:
    df = pd.read_csv(KR_PATH)
    roles = [r for r in ROLE_ORDER if r != "game"]
    tiers = list(AI_TIERS.keys())

    matrix = pd.DataFrame(0.0, index=[ROLE_LABELS[r] for r in roles],
                           columns=[TIER_LABELS[t] for t in tiers])
    for role in roles:
        sub = df[df["role"] == role]
        n = len(sub)
        if n == 0:
            continue
        for tier_key in tiers:
            members = set(AI_TIERS[tier_key])
            count = sub["skills"].fillna("").apply(
                lambda s: bool(set(s.split("|")) & members) if s else False
            ).sum()
            matrix.loc[ROLE_LABELS[role], TIER_LABELS[tier_key]] = count / n * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(matrix.columns, rotation=15, ha="right", fontsize=10)
    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(matrix.index, fontsize=10)
    ax.set_title("RQ1-H: 직무군 × AI 4-tier — 어떤 직무가 어떤 AI를 요구하나 (% 공고)", fontsize=13)

    for i in range(len(roles)):
        for j in range(len(tiers)):
            v = matrix.values[i, j]
            color = "white" if v > 55 else "black"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="해당 tier 보유 공고 비율 (%)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kr_role_x_ai_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ1-H] 저장:", f"{FIG_DIR}/kr_role_x_ai_tier.png")
    print("\n매트릭스 (% 공고):")
    print(matrix.round(1).to_string())


# ---------------------------------------------------------------------------
# RQ2. LinkedIn 글로벌 분석 — A 상위스킬 / B 4-tier 보유율 / B-2 상호배타 / H 직무×Tier
# ---------------------------------------------------------------------------
def _load_linkedin() -> pd.DataFrame | None:
    if not os.path.exists(LINKEDIN_PATH):
        print(f"[RQ2] 스킵 — {LINKEDIN_PATH} 없음 (02 미실행)")
        return None
    return pd.read_csv(LINKEDIN_PATH)


def rq2_top_skills() -> None:
    df = _load_linkedin()
    if df is None:
        return
    token_lists = load_token_lists(df, "skills")
    freq = token_frequency(token_lists)
    print(f"[RQ2-A] 글로벌 LinkedIn {len(df):,}건 (2024-04 단면) · 유니크 스킬 {len(freq):,}개 · "
          f"상위 5 → {freq.head(5).to_dict()}")

    top20 = freq.head(20)
    colors = [CAT_COLORS[category_of(s)] for s in top20.index]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.set_title(f"RQ2-A: 글로벌 LinkedIn 상위 20 기술스택 ({len(df):,}건 · 2024-04 단면)", fontsize=13)
    ax.set_xlabel("언급 공고 수")
    for bar, v in zip(bars, top20.values[::-1]):
        ax.text(v + max(top20) * 0.005, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/linkedin_top_skills.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ2-A] 저장:", f"{FIG_DIR}/linkedin_top_skills.png")


def rq2_ai_tier_breakdown() -> dict:
    df = _load_linkedin()
    if df is None:
        return {}
    n_total = len(df)
    tier_counts = {t: 0 for t in AI_TIERS}
    for s in df["skills"].fillna(""):
        tokens = set(s.split("|")) if s else set()
        for tier_key in AI_TIERS:
            if tokens & set(AI_TIERS[tier_key]):
                tier_counts[tier_key] += 1

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [TIER_LABELS[t] for t in AI_TIERS]
    counts = [tier_counts[t] for t in AI_TIERS]
    colors = [TIER_COLORS[t] for t in AI_TIERS]
    bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1])
    ax.set_title(f"RQ2-B: 글로벌 AI 4-tier 보유율 (중복 허용 · LinkedIn {n_total:,}건)", fontsize=13)
    ax.set_xlabel("해당 tier 토큰 1개 이상 포함한 공고 수")
    for bar, c in zip(bars, counts[::-1]):
        ax.text(c + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{c}건 ({c/n_total*100:.1f}%)", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/linkedin_ai_tier_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ2-B] 저장:", f"{FIG_DIR}/linkedin_ai_tier_breakdown.png")
    for t, c in tier_counts.items():
        print(f"  {TIER_LABELS[t]:30s} {c:5d}건 ({c/n_total*100:5.1f}%) [중복 허용]")
    return tier_counts


def rq2_ai_exclusive_mode() -> dict:
    df = _load_linkedin()
    if df is None:
        return {}
    n_total = len(df)
    members_by_tier = {t: set(AI_TIERS[t]) for t in AI_TIERS}

    mode_counts = {"Tier A · AI 코딩 도구": 0, "Tier B · AI 모델·플랫폼": 0,
                   "Tier C · AI/ML 직무 역량": 0, "Tier D · AI 일반어(광의)만": 0,
                   "AI 미언급": 0}
    mode_colors = {"Tier A · AI 코딩 도구": TIER_COLORS["tier_a_coding_tool"],
                   "Tier B · AI 모델·플랫폼": TIER_COLORS["tier_b_model_platform"],
                   "Tier C · AI/ML 직무 역량": TIER_COLORS["tier_c_ml_skill"],
                   "Tier D · AI 일반어(광의)만":   TIER_COLORS["tier_d_generic"],
                   "AI 미언급": "#e0e0e0"}

    for s in df["skills"].fillna(""):
        tokens = set(s.split("|")) if s else set()
        if tokens & members_by_tier["tier_a_coding_tool"]:
            mode_counts["Tier A · AI 코딩 도구"] += 1
        elif tokens & members_by_tier["tier_b_model_platform"]:
            mode_counts["Tier B · AI 모델·플랫폼"] += 1
        elif tokens & members_by_tier["tier_c_ml_skill"]:
            mode_counts["Tier C · AI/ML 직무 역량"] += 1
        elif tokens & members_by_tier["tier_d_generic"]:
            mode_counts["Tier D · AI 일반어(광의)만"] += 1
        else:
            mode_counts["AI 미언급"] += 1
    assert sum(mode_counts.values()) == n_total, "상호배타 분류 합계 불일치"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = list(mode_counts.keys())
    counts = list(mode_counts.values())
    colors = [mode_colors[l] for l in labels]
    bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1])
    ax.set_title(f"RQ2-B-2: 글로벌 AI 사용 모드 (상호배타 · LinkedIn {n_total:,}건 · 2024-04)",
                 fontsize=13)
    ax.set_xlabel("공고 수 (합계 = 총 공고 수)")
    for bar, c in zip(bars, counts[::-1]):
        ax.text(c + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{c}건 ({c/n_total*100:.1f}%)", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/linkedin_ai_exclusive_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ2-B-2] 저장:", f"{FIG_DIR}/linkedin_ai_exclusive_mode.png  [상호배타 검증 통과]")
    for m, c in mode_counts.items():
        print(f"  {m:30s} {c:5d}건 ({c/n_total*100:5.1f}%)")
    return mode_counts


def rq2_role_x_ai_tier() -> None:
    df = _load_linkedin()
    if df is None:
        return
    roles = [r for r in ROLE_ORDER if r not in ("game",)]  # game 표본 적어 제외
    tiers = list(AI_TIERS.keys())

    matrix = pd.DataFrame(0.0, index=[ROLE_LABELS[r] for r in roles],
                           columns=[TIER_LABELS[t] for t in tiers])
    for role in roles:
        sub = df[df["role"] == role]
        n = len(sub)
        if n == 0:
            continue
        for tier_key in tiers:
            members = set(AI_TIERS[tier_key])
            count = sub["skills"].fillna("").apply(
                lambda s: bool(set(s.split("|")) & members) if s else False
            ).sum()
            matrix.loc[ROLE_LABELS[role], TIER_LABELS[tier_key]] = count / n * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(matrix.columns, rotation=15, ha="right", fontsize=10)
    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(matrix.index, fontsize=10)
    ax.set_title(f"RQ2-H: [글로벌] 직무군 × AI 4-tier (% 공고 · LinkedIn 2024-04)", fontsize=13)
    for i in range(len(roles)):
        for j in range(len(tiers)):
            v = matrix.values[i, j]
            color = "white" if v > 55 else "black"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="해당 tier 보유 공고 비율 (%)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/linkedin_role_x_ai_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[RQ2-H] 저장:", f"{FIG_DIR}/linkedin_role_x_ai_tier.png")
    print("\n매트릭스 (% 공고):")
    print(matrix.round(1).to_string())


# ---------------------------------------------------------------------------
# RQ3. 뉴스 담론 시계열 (2021–2026)
# ---------------------------------------------------------------------------
def _load_news() -> pd.DataFrame | None:
    if not os.path.exists(NEWS_PATH):
        print(f"[RQ3] 스킵 — {NEWS_PATH} 없음 (01 미실행)")
        return None
    df = pd.read_csv(NEWS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _chatgpt_axvline(ax, label_y: float = 0.95) -> None:
    ax.axvline(CHATGPT_LAUNCH, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(CHATGPT_LAUNCH, ax.get_ylim()[1] * label_y, "  ChatGPT 출시",
            color="gray", fontsize=9, va="top")


def rq3_news_volume() -> None:
    """RQ3-A. 월별 기사 수 추세."""
    df = _load_news()
    if df is None:
        return
    monthly = monthly_count(df)
    monthly.index = pd.to_datetime(monthly.index)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly.index, monthly.values, color="steelblue", linewidth=1.6)
    ax.fill_between(monthly.index, monthly.values, color="steelblue", alpha=0.15)
    ax.set_title(f"RQ3-A: 월별 뉴스 기사 수 ({len(df):,}건 · 2021-01 ~ 2026-05)", fontsize=13)
    ax.set_ylabel("기사 수")
    _chatgpt_axvline(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_monthly_volume.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-A] 저장: {FIG_DIR}/news_monthly_volume.png · 월평균 {monthly.mean():.0f}건")


def rq3_ai_tier_trend() -> None:
    """RQ3-B. AI 4-tier 월별 언급률(%)."""
    df = _load_news()
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for tier in TIER_LABELS:
        s = monthly_tier_share(df, tier_col=tier)
        s.index = pd.to_datetime(s.index)
        s_smooth = s.rolling(3, min_periods=1, center=True).mean()
        ax.plot(s_smooth.index, s_smooth.values,
                color=TIER_COLORS[tier], label=TIER_LABELS[tier], linewidth=1.8)
    ax.set_title("RQ3-B: AI 4-tier 월별 언급률 (3개월 이동평균, % of articles)", fontsize=13)
    ax.set_ylabel("언급률 (%)")
    ax.legend(loc="upper left", fontsize=10)
    _chatgpt_axvline(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_ai_tier_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-B] 저장: {FIG_DIR}/news_ai_tier_trend.png")

    # before/after 평균 (요약 출력)
    print("  before/after 평균 언급률(%):")
    for tier in TIER_LABELS:
        b = df[df["period"] == "before"][tier].mean() * 100
        a = df[df["period"] == "after"][tier].mean() * 100
        print(f"    {tier:30s}  before {b:5.1f} → after {a:5.1f}  (Δ +{a-b:+.1f}pp)")


def rq3_ai_tool_emergence() -> None:
    """RQ3-C. 주요 AI 도구·모델 시계열 (Copilot·Cursor·Claude Code·ChatGPT·GPT·Claude·Gemini)."""
    df = _load_news()
    if df is None:
        return

    targets = [
        ("ChatGPT", "#d62728"),
        ("GPT-4",   "#ff7f0e"),
        ("Claude",  "#9467bd"),
        ("Gemini",  "#1f77b4"),
        ("Copilot", "#2ca02c"),
        ("Cursor",  "#8c564b"),
        ("Claude Code", "#e377c2"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    for skill, color in targets:
        s = monthly_skill_rate(df, skill=skill)
        s.index = pd.to_datetime(s.index)
        s_smooth = s.rolling(3, min_periods=1, center=True).mean()
        ax.plot(s_smooth.index, s_smooth.values,
                color=color, label=skill, linewidth=1.6)
    ax.set_title("RQ3-C: 주요 AI 도구·모델 월별 언급률 (3개월 이동평균)", fontsize=13)
    ax.set_ylabel("언급률 (%)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    _chatgpt_axvline(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_ai_tool_emergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-C] 저장: {FIG_DIR}/news_ai_tool_emergence.png")


def rq3_framing_trend() -> None:
    """RQ3-D. 채용 담론 프레임 시계열 (위기·대체·재편·기회).

    프레임 키워드는 명사뿐 아니라 동사형('사라지', '늘어')도 포함하므로
    토큰이 아닌 원본 body 텍스트에서 substring 검색이 필요. 01 단계에서
    body는 csv에 저장하지 않으므로 원본 xlsx에서 다시 로드해 join.
    """
    df = _load_news()
    if df is None:
        return

    # raw xlsx에서 body만 가져와 join (news_id 키)
    from src.collection.bigkinds_news import load_news as _load_raw
    raw = _load_raw(ok_only=False)[["news_id", "body"]]
    raw["news_id"] = raw["news_id"].astype(str)
    df["news_id"] = df["news_id"].astype(str)
    df = df.merge(raw, on="news_id", how="left")
    df["body"] = df["body"].fillna("")

    frame_colors = {
        "crisis": "#d62728", "replace": "#9467bd",
        "restructure": "#1f77b4", "opportunity": "#2ca02c",
    }
    frame_labels = {
        "crisis": "위기·불안", "replace": "대체·소멸",
        "restructure": "재편·전환", "opportunity": "기회·성장",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for frame, color in frame_colors.items():
        s = monthly_frame_intensity(df, frame=frame, text_col="body")
        s.index = pd.to_datetime(s.index)
        s_smooth = s.rolling(3, min_periods=1, center=True).mean()
        ax.plot(s_smooth.index, s_smooth.values,
                color=color, label=frame_labels[frame], linewidth=1.8)
    ax.set_title("RQ3-D: 채용 담론 프레임 월별 강도 (기사당 키워드 등장 횟수, 3개월 이동평균)", fontsize=12)
    ax.set_ylabel("평균 등장 횟수 / 기사")
    ax.legend(loc="upper left", fontsize=10)
    _chatgpt_axvline(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_framing_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-D] 저장: {FIG_DIR}/news_framing_trend.png")

    # before/after 평균
    print("  before/after 프레임 강도 (기사당 평균):")
    for frame in frame_labels:
        kws = FRAMING_KEYWORDS[frame]
        b = df[df["period"] == "before"]["body"].fillna("").apply(
            lambda t: sum(t.count(k) for k in kws)).mean()
        a = df[df["period"] == "after"]["body"].fillna("").apply(
            lambda t: sum(t.count(k) for k in kws)).mean()
        print(f"    {frame_labels[frame]:8s}  before {b:5.2f} → after {a:5.2f}  (Δ {a-b:+.2f})")


def rq3_period_wordcloud() -> None:
    """RQ3-E. before/after 워드클라우드 (본문 토큰 빈도 기반)."""
    from wordcloud import WordCloud
    from src.visualization._fonts import get_font_path

    df = _load_news()
    if df is None:
        return

    before_freq = period_token_freq(df, period="before", top_n=120)
    after_freq = period_token_freq(df, period="after", top_n=120)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, freq, label in zip(
        axes,
        [before_freq, after_freq],
        [f"BEFORE ChatGPT ({(df['period']=='before').sum():,}건)",
         f"AFTER ChatGPT ({(df['period']=='after').sum():,}건)"],
    ):
        wc = WordCloud(
            font_path=get_font_path(),
            background_color="white",
            width=900, height=500, max_words=120,
        ).generate_from_frequencies(freq.to_dict())
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    fig.suptitle("RQ3-E: 뉴스 본문 토큰 빈도 — ChatGPT 출시 전·후 (Top 120)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_period_wordcloud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-E] 저장: {FIG_DIR}/news_period_wordcloud.png")
    print(f"  before 상위 10: {before_freq.head(10).to_dict()}")
    print(f"  after 상위 10:  {after_freq.head(10).to_dict()}")


# ─── 뉴스 전체 데이터 캐시 (body + 직무 라벨 포함) ──────────────────
_NEWS_FULL_CACHE: dict = {}


def _load_news_full() -> pd.DataFrame | None:
    """processed csv + raw body + 직무 멀티라벨 — 모듈 내 캐시."""
    if "df" in _NEWS_FULL_CACHE:
        return _NEWS_FULL_CACHE["df"]
    df = _load_news()
    if df is None:
        return None
    print("[init] body + 직무 라벨 로드 중 (~60초)...")
    from src.collection.bigkinds_news import load_news as _load_raw
    raw = _load_raw(ok_only=False)[["news_id", "body"]]
    raw["news_id"] = raw["news_id"].astype(str)
    df["news_id"] = df["news_id"].astype(str)
    df = df.merge(raw, on="news_id", how="left")
    df["body"] = df["body"].fillna("")
    text = df["title"].fillna("") + " " + df["body"]
    roles_series = text.apply(match_all_roles)
    for role in ROLE_LABELS:
        df[f"role_{role}"] = roles_series.apply(lambda s, r=role: int(r in s))
    _NEWS_FULL_CACHE["df"] = df
    return df


_ROLE_PALETTE = {
    "ai_ml":    "#d62728",  # red
    "backend":  "#1f77b4",  # blue
    "frontend": "#17becf",  # cyan
    "fullstack":"#9467bd",  # purple
    "mobile":   "#e377c2",  # pink
    "data":     "#2ca02c",  # green
    "devops":   "#ff7f0e",  # orange
    "embedded": "#7f7f7f",  # gray
    "security": "#8c564b",  # brown
    "qa":       "#bcbd22",  # olive
    "game":     "#aec7e8",  # lightblue
}


def rq3_role_trend() -> None:
    """RQ3-G. 직무별 월별 언급률 — 2패널 (상: AI/ML·데이터·인프라계, 하: 웹·앱 개발)."""
    df = _load_news_full()
    if df is None:
        return

    GROUPS = [
        ("AI/ML · 데이터 · 인프라 · 보안 · QA · 게임",
         ["ai_ml", "data", "devops", "security", "embedded", "qa", "game"]),
        ("웹·앱 개발 (백엔드 · 프론트엔드 · 풀스택 · 모바일)",
         ["backend", "frontend", "fullstack", "mobile"]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9.5), sharex=True)
    for ax, (panel_title, roles) in zip(axes, GROUPS):
        for role in roles:
            s = df.groupby("month")[f"role_{role}"].mean() * 100
            s.index = pd.to_datetime(s.index)
            s_smooth = s.rolling(3, min_periods=1, center=True).mean()
            ax.plot(s_smooth.index, s_smooth.values,
                    color=_ROLE_PALETTE[role],
                    label=ROLE_LABELS[role], linewidth=1.8)
        ax.set_title(panel_title, fontsize=12)
        ax.set_ylabel("언급률 (%)")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        _chatgpt_axvline(ax)
    fig.suptitle("RQ3-G: 직무별 월별 뉴스 언급률 (3개월 이동평균)", fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_role_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-G] 저장: {FIG_DIR}/news_role_trend.png")
    print("  before / after 평균 언급률 — 11개 직무 (%):")
    all_roles = sum((g[1] for g in GROUPS), [])
    for role in all_roles:
        b = df[df["period"] == "before"][f"role_{role}"].mean() * 100
        a = df[df["period"] == "after"][f"role_{role}"].mean() * 100
        print(f"    {ROLE_LABELS[role]:14s}  before {b:5.1f} → after {a:5.1f}  (Δ {a-b:+5.1f}pp)")


def rq3_role_groups() -> None:
    """RQ3-G-2. 직무 그룹 묶음 시계열 — 웹·앱 개발을 한 시점에 비교."""
    df = _load_news_full()
    if df is None:
        return

    GROUP_DEF = {
        "AI/ML":                            ["ai_ml"],
        "웹 개발 (BE+FE+풀스택)":            ["backend", "frontend", "fullstack"],
        "모바일 앱":                          ["mobile"],
        "데이터":                            ["data"],
        "인프라·보안 (DevOps+임베디드+보안)":   ["devops", "embedded", "security"],
        "QA·게임":                           ["qa", "game"],
    }
    COLORS = {
        "AI/ML": "#d62728",
        "웹 개발 (BE+FE+풀스택)": "#1f77b4",
        "모바일 앱": "#e377c2",
        "데이터": "#2ca02c",
        "인프라·보안 (DevOps+임베디드+보안)": "#ff7f0e",
        "QA·게임": "#bcbd22",
    }

    # 그룹 boolean 컬럼 (OR)
    for gname, roles in GROUP_DEF.items():
        cols = [f"role_{r}" for r in roles]
        df[f"grp_{gname}"] = df[cols].max(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for gname in GROUP_DEF:
        s = df.groupby("month")[f"grp_{gname}"].mean() * 100
        s.index = pd.to_datetime(s.index)
        s_smooth = s.rolling(3, min_periods=1, center=True).mean()
        ax.plot(s_smooth.index, s_smooth.values,
                color=COLORS[gname], label=gname, linewidth=1.9)
    ax.set_title("RQ3-G-2: 직무 그룹 묶음 월별 언급률 (3개월 이동평균)", fontsize=13)
    ax.set_ylabel("언급률 (%)")
    ax.legend(loc="upper left", fontsize=9)
    _chatgpt_axvline(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_role_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-G-2] 저장: {FIG_DIR}/news_role_groups.png")
    print("  before / after 평균 (%):")
    for gname in GROUP_DEF:
        b = df[df["period"] == "before"][f"grp_{gname}"].mean() * 100
        a = df[df["period"] == "after"][f"grp_{gname}"].mean() * 100
        print(f"    {gname:32s}  before {b:5.1f} → after {a:5.1f}  (Δ {a-b:+5.1f}pp)")


def rq3_role_x_tier() -> None:
    """RQ3-H. 직무 × AI Tier 매트릭스 — before vs after 2패널 heatmap.

    셀 값 = (해당 직무 매칭 기사 중 그 tier도 매칭한 비율, %).
    """
    import numpy as np
    df = _load_news_full()
    if df is None:
        return

    role_totals = {r: df[f"role_{r}"].sum() for r in ROLE_LABELS if r != "other"}
    top_roles = [r for r, _ in sorted(role_totals.items(), key=lambda x: -x[1])[:7]]
    tiers = list(TIER_LABELS.keys())

    def build(sub: pd.DataFrame) -> pd.DataFrame:
        mat = pd.DataFrame(index=[ROLE_LABELS[r] for r in top_roles],
                           columns=[TIER_LABELS[t] for t in tiers], dtype=float)
        for role in top_roles:
            role_mask = sub[f"role_{role}"] == 1
            n = role_mask.sum()
            for tier in tiers:
                if n > 0:
                    mat.loc[ROLE_LABELS[role], TIER_LABELS[tier]] = sub.loc[role_mask, tier].mean() * 100
                else:
                    mat.loc[ROLE_LABELS[role], TIER_LABELS[tier]] = 0
        return mat

    mat_before = build(df[df["period"] == "before"])
    mat_after = build(df[df["period"] == "after"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    vmax = max(mat_before.values.max(), mat_after.values.max())
    for ax, mat, label in zip(axes, [mat_before, mat_after],
                                ["BEFORE ChatGPT (2021-01 ~ 2022-11)",
                                 "AFTER ChatGPT (2022-12 ~ 2026-05)"]):
        im = ax.imshow(mat.values, cmap="OrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([TIER_LABELS[t].replace(" · ", "\n") for t in tiers], fontsize=9)
        ax.set_yticks(range(len(top_roles)))
        ax.set_yticklabels([ROLE_LABELS[r] for r in top_roles], fontsize=10)
        ax.set_title(label, fontsize=12)
        for i in range(len(top_roles)):
            for j in range(len(tiers)):
                v = mat.values[i, j]
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")
    fig.suptitle("RQ3-H: 직무 × AI Tier 결합 매트릭스 (해당 직무 기사 중 그 tier도 매칭한 비율)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_role_x_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-H] 저장: {FIG_DIR}/news_role_x_tier.png")
    print("\n[AFTER] 직무 × Tier (%):")
    print(mat_after.round(1).to_string())


def rq3_career_stage_trend() -> None:
    """RQ3-I. 신입·주니어·경력·시니어 채용 담론 시계열."""
    df = _load_news_full()
    if df is None:
        return

    STAGE_KEYWORDS = {
        "신입":     ["신입 ", "신입사원", "신입 개발자", "신입 채용"],
        "주니어":   ["주니어", "junior", "Junior", "JUNIOR"],
        "경력":     ["경력직", "경력자", "경력 개발자", "경력 채용", "경력 우대"],
        "시니어":   ["시니어", "senior", "Senior", "SENIOR", "선임 개발자"],
    }
    STAGE_COLORS = {"신입": "#1f77b4", "주니어": "#2ca02c",
                    "경력": "#ff7f0e", "시니어": "#d62728"}

    for stage, kws in STAGE_KEYWORDS.items():
        df[f"stage_{stage}"] = df["body"].apply(
            lambda t: int(any(k in t for k in kws)) if isinstance(t, str) else 0
        )

    fig, ax = plt.subplots(figsize=(12, 5))
    for stage in STAGE_KEYWORDS:
        s = df.groupby("month")[f"stage_{stage}"].mean() * 100
        s.index = pd.to_datetime(s.index)
        s_smooth = s.rolling(3, min_periods=1, center=True).mean()
        ax.plot(s_smooth.index, s_smooth.values,
                color=STAGE_COLORS[stage], label=stage, linewidth=1.8)
    ax.set_title("RQ3-I: 경력 단계별 채용 담론 월별 언급률 (3개월 이동평균)", fontsize=13)
    ax.set_ylabel("언급률 (%)")
    ax.legend(loc="upper left", fontsize=10)
    _chatgpt_axvline(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_career_stage_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-I] 저장: {FIG_DIR}/news_career_stage_trend.png")
    print("  before / after 평균 언급률 (%):")
    for stage in STAGE_KEYWORDS:
        b = df[df["period"] == "before"][f"stage_{stage}"].mean() * 100
        a = df[df["period"] == "after"][f"stage_{stage}"].mean() * 100
        print(f"    {stage:8s}  before {b:5.1f} → after {a:5.1f}  (Δ {a-b:+5.1f}pp)")


def rq3_yearly_tier_summary() -> None:
    """RQ3-F. 연도별 AI 4-tier 보유율 — 월별 노이즈 줄인 요약 막대."""
    df = _load_news()
    if df is None:
        return

    rows = {tier: yearly_tier_share(df, tier_col=tier) for tier in TIER_LABELS}
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(table.index))
    width = 0.20
    for i, tier in enumerate(TIER_LABELS):
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], table[tier].values, width,
               color=TIER_COLORS[tier], label=TIER_LABELS[tier])
    ax.set_xticks(list(x))
    ax.set_xticklabels(table.index)
    ax.set_ylabel("언급률 (% of articles)")
    ax.set_title("RQ3-F: 연도별 AI 4-tier 보유율", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/news_yearly_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3-F] 저장: {FIG_DIR}/news_yearly_tier.png")
    print("\n연도별 tier 보유율 (%):")
    print(table.round(1).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    freq_kr = rq1_kr_jobs_top_skills()
    rq1_ai_tier_breakdown()
    rq1_ai_exclusive_mode()
    rq1_ai_tier_detail()
    rq1_ai_overlap()
    rq1_kr_jobs_cooccurrence()
    rq1_wordcloud(freq_kr)
    rq1_role_distribution()
    rq1_top_skills_by_role()
    rq1_role_x_ai_tier()
    rq2_top_skills()
    rq2_ai_tier_breakdown()
    rq2_ai_exclusive_mode()
    rq2_role_x_ai_tier()
    rq3_news_volume()
    rq3_ai_tier_trend()
    rq3_ai_tool_emergence()
    rq3_framing_trend()
    rq3_period_wordcloud()
    rq3_yearly_tier_summary()
    rq3_role_trend()
    rq3_role_groups()
    rq3_role_x_tier()
    rq3_career_stage_trend()
