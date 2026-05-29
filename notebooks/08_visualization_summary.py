"""
08_visualization_summary.py
담당: 미정

04~07 스크립트에서 생성된 outputs/figures/ 파일을 확인하고
발표자료용 최종 정리를 수행한다.
"""

import os
from pathlib import Path

FIG_DIR = "outputs/figures"

EXPECTED_FIGURES = {
    "kr_jobs_top_skills.png":         "RQ1: 국내공고 상위 기술스택 막대그래프",
    "linkedin_top_skills.png":        "RQ2: LinkedIn 상위 기술스택 막대그래프",
    "news_ai_trend.png":              "RQ3: 뉴스 AI 키워드 월별 언급률 시계열",
    "wc_kr_jobs.png":                 "WordCloud — 국내공고",
    "wc_news_before.png":             "WordCloud — 뉴스 before ChatGPT",
    "wc_news_after.png":              "WordCloud — 뉴스 after ChatGPT",
    "cooccurrence_network.png":       "Co-occurrence Network (수업 외)",
    "bertopic_topics.html":           "BERTopic 토픽 분포 (수업 외)",
    "umap_clusters.png":              "UMAP 직무군 산점도 (수업 외)",
    "confusion_matrix.png":           "직무군 분류 Confusion Matrix",
    "rq4_discourse_vs_reality.png":   "RQ4 핵심: 담론 vs 현실 비교",
    "rq4_domestic_vs_global.png":     "국내 vs 글로벌 기술스택 비교",
}


def check_outputs() -> None:
    print("=== 산출 시각화 체크리스트 ===\n")
    all_ok = True
    for fname, desc in EXPECTED_FIGURES.items():
        path = Path(FIG_DIR) / fname
        status = "O" if path.exists() else "X (미생성)"
        if not path.exists():
            all_ok = False
        print(f"  [{status}] {fname}")
        print(f"         {desc}")
    print()
    if all_ok:
        print("모든 시각화 파일 생성 완료.")
    else:
        print("미생성 파일이 있습니다. 해당 스크립트를 먼저 실행하세요.")


if __name__ == "__main__":
    check_outputs()
