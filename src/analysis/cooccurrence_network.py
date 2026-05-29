"""
Co-occurrence Network 분석 (수업 외 방법론)
기술스택 동시출현 행렬 → NetworkX 그래프 구축
예: Java–Spring–MySQL, AI–Python–PyTorch
"""

import itertools
from collections import Counter
import networkx as nx
import pandas as pd


def build_cooccurrence_matrix(token_lists: list[list[str]], min_count: int = 5) -> pd.DataFrame:
    """토큰 리스트 집합에서 동시출현 행렬 생성."""
    pair_counter: Counter = Counter()
    for tokens in token_lists:
        unique_tokens = list(set(tokens))
        for pair in itertools.combinations(sorted(unique_tokens), 2):
            pair_counter[pair] += 1
    pairs = [(a, b, c) for (a, b), c in pair_counter.items() if c >= min_count]
    return pd.DataFrame(pairs, columns=["source", "target", "weight"])


def build_graph(edge_df: pd.DataFrame) -> nx.Graph:
    """엣지 DataFrame → NetworkX 그래프."""
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    return G
