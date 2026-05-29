"""
Word2Vec 공기어 분석 (수업 외 방법론)
특정 키워드(AI, LLM, 신입, React 등)와 의미적으로 가까운 단어 탐색
"""

from gensim.models import Word2Vec
import pandas as pd


def train_word2vec(token_lists: list[list[str]], vector_size: int = 100, window: int = 5, min_count: int = 5) -> Word2Vec:
    """Word2Vec 모델 학습."""
    model = Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
    )
    return model


def most_similar(model: Word2Vec, word: str, topn: int = 10) -> list[tuple]:
    """주어진 단어와 가장 유사한 단어 목록 반환."""
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return []
