"""
UMAP 임베딩 2차원 시각화 (수업 외 방법론)
Sentence-BERT 임베딩 → UMAP 축소 → 직무군 분리 확인

reduce_embeddings() / plot_2d() 로 축소와 시각화를 분리해, 호출부에서
파라미터 재시도(n_neighbors/min_dist)와 분리도 측정을 할 수 있게 한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap


def reduce_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """임베딩을 UMAP 2D 좌표로 축소."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def plot_2d(
    coords: np.ndarray,
    labels: list[str],
    title: str = "UMAP — Job Category Clusters",
    save_path: str | None = None,
    alpha: float = 0.5,
    s: int = 8,
) -> None:
    """2D 좌표를 라벨별 색상 산점도로 그린다."""
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
    plt.figure(figsize=(9, 7))
    for label, group in df.groupby("label"):
        plt.scatter(group["x"], group["y"], label=label, s=s, alpha=alpha)
    plt.legend(markerscale=3, fontsize=9)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def reduce_and_plot(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "UMAP — Job Category Clusters",
    save_path: str | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """축소 + 시각화 한 번에 (기존 호출 호환)."""
    coords = reduce_embeddings(embeddings, n_neighbors, min_dist, random_state)
    plot_2d(coords, labels, title=title, save_path=save_path)
