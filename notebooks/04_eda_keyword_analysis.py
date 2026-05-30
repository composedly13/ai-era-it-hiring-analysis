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
from src.analysis.time_series import monthly_keyword_count
from src.analysis.cooccurrence_network import build_cooccurrence_matrix, build_graph
from src.visualization.wordcloud_gen import generate_wordcloud
from src.visualization.network_plot import draw_network
from src.preprocessing.tech_dictionary import AI_TIERS, TECH_DICT, get_ai_tier
from src.preprocessing.role_classifier import ROLE_LABELS

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
    "tier_d_generic":       "Tier D · 모호한 'AI'",
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
                   "Tier C · AI/ML 직무 역량": 0, "Tier D · 모호한 'AI'만": 0,
                   "AI 미언급": 0}
    mode_colors = {"Tier A · AI 코딩 도구": TIER_COLORS["tier_a_coding_tool"],
                   "Tier B · AI 모델·플랫폼": TIER_COLORS["tier_b_model_platform"],
                   "Tier C · AI/ML 직무 역량": TIER_COLORS["tier_c_ml_skill"],
                   "Tier D · 모호한 'AI'만":   TIER_COLORS["tier_d_generic"],
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
            mode_counts["Tier D · 모호한 'AI'만"] += 1
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
                   "Tier C · AI/ML 직무 역량": 0, "Tier D · 모호한 'AI'만": 0,
                   "AI 미언급": 0}
    mode_colors = {"Tier A · AI 코딩 도구": TIER_COLORS["tier_a_coding_tool"],
                   "Tier B · AI 모델·플랫폼": TIER_COLORS["tier_b_model_platform"],
                   "Tier C · AI/ML 직무 역량": TIER_COLORS["tier_c_ml_skill"],
                   "Tier D · 모호한 'AI'만":   TIER_COLORS["tier_d_generic"],
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
            mode_counts["Tier D · 모호한 'AI'만"] += 1
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
# RQ3. 뉴스 AI 키워드 월별 언급률 시계열
# ---------------------------------------------------------------------------
def rq3_news_timeseries() -> None:
    if not os.path.exists(NEWS_PATH):
        print(f"[RQ3] 스킵 — {NEWS_PATH} 없음 (01 미실행)")
        return
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
    plt.close(fig)
    print("[RQ3] 저장:", f"{FIG_DIR}/news_ai_trend.png")


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
    rq3_news_timeseries()
