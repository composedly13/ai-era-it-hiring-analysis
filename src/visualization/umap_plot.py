"""
UMAP 임베딩 2차원 시각화 (수업 외 방법론)
Sentence-BERT 임베딩 → UMAP 축소 → 직무군 분리 확인
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap


def reduce_and_plot(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "UMAP — Job Category Clusters",
    save_path: str | None = None,
) -> None:
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    df = pd.DataFrame({"x": embedding_2d[:, 0], "y": embedding_2d[:, 1], "label": labels})
    for label, group in df.groupby("label"):
        plt.scatter(group["x"], group["y"], label=label, s=5, alpha=0.5)

    plt.legend(markerscale=3, fontsize=8)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
