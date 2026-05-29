"""Co-occurrence Network 시각화."""
import networkx as nx
import matplotlib.pyplot as plt


def draw_network(G: nx.Graph, title: str = "", save_path: str | None = None, top_n: int = 50) -> None:
    """상위 N개 엣지만 그려 시각화 단순화."""
    # 가중치 기준 상위 엣지 추출
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 1), reverse=True)[:top_n]
    sub_nodes = set()
    for u, v, _ in edges:
        sub_nodes.update([u, v])
    subG = G.subgraph(sub_nodes)

    pos = nx.spring_layout(subG, seed=42, k=0.5)
    weights = [subG[u][v]["weight"] for u, v in subG.edges()]
    nx.draw_networkx(subG, pos, width=[w / max(weights) * 3 for w in weights], node_size=600, font_size=8)

    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
