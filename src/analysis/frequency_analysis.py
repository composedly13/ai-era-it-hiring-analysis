"""빈도 분석 — 전체 기술스택 분포 파악."""
import pandas as pd
from collections import Counter


def token_frequency(token_lists: list[list[str]]) -> pd.Series:
    """토큰 리스트 집합에서 빈도 계산."""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    return pd.Series(counter).sort_values(ascending=False)
