"""
담론 vs 현실 비교 차트 (RQ4 핵심 시각화)
뉴스 키워드 빈도 vs 국내공고 채용공고 키워드 빈도 비교표/막대차트
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_discourse_vs_reality(
    news_freq: pd.Series,
    kr_jobs_freq: pd.Series,
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """뉴스 담론과 국내공고 요구역량 상위 N개 나란히 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    news_top = news_freq.head(top_n)
    kr_jobs_top = kr_jobs_freq.head(top_n)

    axes[0].barh(news_top.index[::-1], news_top.values[::-1], color="#4c72b0")
    axes[0].set_title("뉴스 담론 상위 키워드 (빅카인즈)", fontsize=13)

    axes[1].barh(kr_jobs_top.index[::-1], kr_jobs_top.values[::-1], color="#dd8452")
    axes[1].set_title("실제 채용공고 요구역량 (국내공고)", fontsize=13)

    plt.suptitle("담론 vs 현실: IT 채용 키워드 비교 (RQ4)", fontsize=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
