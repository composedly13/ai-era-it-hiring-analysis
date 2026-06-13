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
    "linkedin_top_skills.png":        "RQ2-A: LinkedIn 글로벌 상위 기술스택",
    "linkedin_ai_tier_breakdown.png": "RQ2-B: 글로벌 AI 4-tier 보유율 (중복 허용)",
    "linkedin_ai_exclusive_mode.png": "RQ2-B-2: 글로벌 AI 사용 모드 (상호배타)",
    "linkedin_role_x_ai_tier.png":    "RQ2-H: 글로벌 직무 × AI 4-tier 히트맵",
    # ---- RQ3 뉴스 담론 시계열 (04 노트북, 01 머지 후) — 총 10개 차트 ----
    "news_monthly_volume.png":        "RQ3-A: 월별 뉴스 기사 수 추세 (×2배 증가)",
    "news_ai_tier_trend.png":         "RQ3-B: AI 4-tier 월별 언급률 (3개월 이동평균)",
    "news_ai_tool_emergence.png":     "RQ3-C: 주요 AI 도구·모델 시계열 (ChatGPT·GPT-4·Claude·Copilot·Gemini)",
    "news_framing_trend.png":         "RQ3-D: 채용 담론 프레임 (위기·대체·재편·기회) 월별 강도",
    "news_period_wordcloud.png":      "RQ3-E: 뉴스 본문 토큰 빈도 — ChatGPT 출시 전·후",
    "news_yearly_tier.png":           "RQ3-F: 연도별 AI 4-tier 보유율 (요약 막대)",
    "news_role_trend.png":            "RQ3-G: 직무별 11개 월별 언급률 (2패널 — AI/인프라계 · 웹·앱 개발)",
    "news_role_groups.png":           "RQ3-G-2: 직무 그룹 묶음 시계열 (웹·앱·인프라·AI/ML)",
    "news_role_x_tier.png":           "RQ3-H: 직무 × AI Tier 결합 매트릭스 (before vs after)",
    "news_career_stage_trend.png":    "RQ3-I: 신입·주니어·경력·시니어 채용 담론 시계열",
    # ---- 보조 분석 / 수업 외 모델링 (05 노트북) ----
    "bertopic_topics.png":            "BERTopic 토픽별 대표 키워드 (논문 4-4-2 / 수업 외)",
    "bertopic_topics.html":           "BERTopic 토픽 분포 인터랙티브 (논문 4-4-2 / 수업 외)",
    "umap_clusters.png":              "UMAP 직무군 산점도 (논문 4-4-3 / 수업 외)",
    # ---- 직무군 분류 (06 노트북) ----
    "confusion_matrix.png":           "직무군 분류 Confusion Matrix (논문 4-4-1 / repo legacy RQ6)",
    # ---- RQ4 담론 vs 현실 (07 노트북) ----
    "rq4_gap_index.png":              "RQ4a: 담론 vs 현실 간극 — 기술스택 순위 사분면 (Spearman)",
    "rq4_tier_gap.png":               "RQ4b: AI 4-tier 담론 vs 현실 — Tier D 'AI' 과잉 / Tier C ML역량 부족",
    "rq4_top_gap_bars.png":           "RQ4c: 담론 과잉·조용한 핵심 Top 15 — GPT 도배 / Docker 침묵",
    "rq4_period_spearman.png":        "RQ4d: 시점별 ρ — before 0.42 → after 0.26 → 2026 0.10 (담론·현실 멀어짐)",
    "rq4_role_x_tier_gap.png":        "RQ4e: 직무 × tier 격차 — Tier D 모든 직무 과잉 / AI·ML Tier C 부족",
    # ---- RQ5 한·글 비교 (07 노트북) ----
    "rq5_quadrant.png":               "RQ5-A: 국내 vs 글로벌 사분면 (% × %)",
    "rq5_ai_tier_compare.png":        "RQ5-B: AI 4-tier 한국 vs 글로벌 비교",
    "rq5_role_x_tier_a_coding_tool.png": "RQ5-C: 직무 × Tier A (코딩 도구) 한·글 비교",
    "rq5_role_x_tier_c_ml_skill.png": "RQ5-C': 직무 × Tier C (ML 역량) 한·글 비교",
    "rq5_key_tokens_gap.png":         "RQ5-D: 핵심 토큰 격차 (한국 - 글로벌, pp)",
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
