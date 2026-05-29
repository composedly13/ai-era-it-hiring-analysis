"""WordCloud 생성 — 발표용 직관 시각화."""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


def generate_wordcloud(freq: Counter | dict, title: str = "", save_path: str | None = None) -> None:
    wc = WordCloud(
        font_path="malgun.ttf",   # 한국어 폰트 (환경에 맞게 변경)
        background_color="white",
        width=1200,
        height=600,
        max_words=100,
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
