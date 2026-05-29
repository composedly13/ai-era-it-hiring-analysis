"""TF-IDF 분석 — 직무군·연도·데이터셋별 변별 키워드 추출."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_tfidf(docs: list[str], max_features: int = 5000) -> tuple:
    """TF-IDF 행렬 및 피처 이름 반환."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(docs)
    return matrix, vectorizer.get_feature_names_out()


def top_terms_per_group(df: pd.DataFrame, text_col: str, group_col: str, top_n: int = 20) -> pd.DataFrame:
    """그룹별 TF-IDF 상위 N개 키워드 반환."""
    # TODO: 구현
    raise NotImplementedError
