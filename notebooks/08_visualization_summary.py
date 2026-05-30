"""
08_visualization_summary.py
담당: 전원

04~07 스크립트에서 생성된 outputs/figures/ 파일을 확인하고
발표자료용 최종 정리를 수행한다.
"""

import os
from pathlib import Path

FIG_DIR = "outputs/figures"

EXPECTED_FIGURES = {
    # ---- RQ1 국내공고 (04 노트북) ----
    "kr_top_skills.png":              "RQ1-A: 국내공고 상위 20 기술스택 (카테고리 색상)",
    "kr_ai_tier_breakdown.png":       "RQ1-B: AI 4-tier 보유율 (중복 허용)",
    "kr_ai_exclusive_mode.png":       "RQ1-B-2: AI 사용 모드 (상호배타 분류)",
    "kr_ai_tier_detail.png":          "RQ1-C: Tier A·C 상세 빈도",
    "kr_ai_overlap.png":              "RQ1-D: 'AI' 1670건의 동반 tier 분포",
    "cooccurrence_network.png":       "RQ1-E: 동시출현 네트워크 (수업 외)",
    "wc_kr_jobs.png":                 "RQ1: WordCloud — 국내공고",
    "kr_role_distribution.png":       "RQ1-F: 직무군 분포",
    "kr_top_skills_by_role.png":      "RQ1-G: 직무군별 상위 7 스킬",
    "kr_role_x_ai_tier.png":          "RQ1-H: 직무 × AI 4-tier 히트맵 (RQ4/5 핵심 자료)",
    # ---- RQ2 LinkedIn (04 노트북, 02 머지 후) ----
    "linkedin_top_skills.png":        "RQ2: LinkedIn 글로벌 상위 기술스택",
    # ---- RQ3 뉴스 (04 노트북, 01 머지 후) ----
    "news_ai_trend.png":              "RQ3: 뉴스 AI 키워드 월별 언급률 시계열",
    "wc_news_before.png":             "WordCloud — 뉴스 before ChatGPT",
    "wc_news_after.png":              "WordCloud — 뉴스 after ChatGPT",
    # ---- 수업 외 모델링 (05 노트북) ----
    "bertopic_topics.html":           "BERTopic 토픽 분포 (수업 외)",
    "umap_clusters.png":              "UMAP 직무군 산점도 (수업 외)",
    # ---- RQ6 분류 모델 (06 노트북) ----
    "confusion_matrix.png":           "직무군 분류 Confusion Matrix",
    # ---- RQ4·RQ5 비교 (07 노트북) ----
    "rq4_gap_index.png":              "RQ4a: 담론 vs 현실 간극지수 (Spearman·순위차)",
    "rq5_domestic_vs_global.png":     "RQ5: 국내 vs 글로벌 기술스택 비교",
}


def check_outputs() -> None:
    print("=== 산출 시각화 체크리스트 ===\n")
    by_phase = {}
    for fname, desc in EXPECTED_FIGURES.items():
        path = Path(FIG_DIR) / fname
        status = "O" if path.exists() else "X"
        by_phase.setdefault(status, []).append((fname, desc))

    present = by_phase.get("O", [])
    missing = by_phase.get("X", [])

    print(f"[O 생성 완료 {len(present)}/{len(EXPECTED_FIGURES)}]")
    for fname, desc in present:
        print(f"  {fname:35s}  {desc}")
    print()
    if missing:
        print(f"[X 미생성 {len(missing)}]")
        for fname, desc in missing:
            print(f"  {fname:35s}  {desc}")
        print("\n해당 노트북을 실행해 산출하세요.")
    else:
        print("전체 산출 완료.")


if __name__ == "__main__":
    check_outputs()
