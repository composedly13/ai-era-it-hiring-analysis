"""Co-occurrence Network 시각화."""
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx


def draw_network(
    G: nx.Graph,
    title: str = "",
    save_path: str | None = None,
    top_n: int = 50,
    node_color_fn: Callable[[str], str] | None = None,
    default_color: str = "#4c72b0",
    legend: dict[str, str] | None = None,
) -> None:
    """상위 N개 엣지만 그려 시각화 단순화.

    node_color_fn: (node_name) -> color hex. 카테고리별 색상.
    legend: {label: color} 매핑이 있으면 우상단에 범례.
    """
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 1), reverse=True)[:top_n]
    subG = G.edge_subgraph([(u, v) for u, v, _ in edges]).copy()

    degrees = dict(subG.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    nodes = list(subG.nodes())
    node_sizes = [800 + (degrees[n] / max_deg) * 2400 for n in nodes]
    node_colors = [node_color_fn(n) if node_color_fn else default_color for n in nodes]

    pos = nx.spring_layout(subG, seed=42, k=1.2, iterations=200)
    weights = [subG[u][v]["weight"] for u, v in subG.edges()]
    max_w = max(weights) if weights else 1

    fig, ax = plt.subplots(figsize=(15, 10))
    nx.draw_networkx_edges(
        subG, pos, ax=ax,
        width=[w / max_w * 4 + 0.3 for w in weights],
        alpha=0.35, edge_color="#888",
    )
    nx.draw_networkx_nodes(
        subG, pos, ax=ax,
        nodelist=nodes,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9, edgecolors="white", linewidths=1.5,
    )
    nx.draw_networkx_labels(subG, pos, ax=ax, font_size=10, font_weight="bold")
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=14)

    if legend:
        patches = [mpatches.Patch(color=c, label=label) for label, c in legend.items()]
        ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.85,
                  title="카테고리", title_fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
