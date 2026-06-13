"""
BERTopic 토픽 모델링 (수업 외 방법론)
Sentence-BERT 임베딩 기반 토픽 군집 + 대표 키워드 (LDA 대비 차별화)

예상 토픽:
  0. Java/Spring 백엔드
  1. React/TS 프론트엔드
  2. Python/SQL 데이터분석
  3. AI/ML 모델개발
  4. 클라우드/DevOps
  5. 보안/시스템운영
  6. 협업/PM
"""

from bertopic import BERTopic
import pandas as pd


def train_bertopic(
    docs: list[str],
    language: str = "multilingual",
    nr_topics=None,
    embedding_model=None,
    vectorizer_model=None,
    min_topic_size: int = 10,
    calculate_probabilities: bool = False,
    embeddings=None,
) -> tuple:
    """BERTopic 모델 학습 및 토픽-문서 매핑 반환.

    - embedding_model: 사전 로드한 SBERT 인스턴스를 넘기면 그것을 사용(UMAP과 임베딩 공유).
      None 이면 language 로 기본 임베딩.
    - vectorizer_model: c-TF-IDF 단어 추출용 CountVectorizer(불용어/min_df/ngram 지정).
    - min_topic_size: 군집 최소 크기. 작을수록 토픽이 잘게 분리됨.
    - embeddings: 사전 계산한 임베딩 배열(중복 계산 회피). docs 와 같은 순서.
    """
    kwargs = dict(
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=calculate_probabilities,
    )
    if embedding_model is not None:
        kwargs["embedding_model"] = embedding_model
    else:
        kwargs["language"] = language
    if vectorizer_model is not None:
        kwargs["vectorizer_model"] = vectorizer_model

    model = BERTopic(**kwargs)
    topics, probs = model.fit_transform(docs, embeddings=embeddings)
    return model, topics, probs


def get_topic_info(model: BERTopic) -> pd.DataFrame:
    """토픽별 대표 키워드 및 문서 수 반환."""
    return model.get_topic_info()
