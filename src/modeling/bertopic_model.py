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


def train_bertopic(docs: list[str], language: str = "multilingual", nr_topics: int = 7) -> tuple:
    """BERTopic 모델 학습 및 토픽-문서 매핑 반환."""
    model = BERTopic(language=language, nr_topics=nr_topics, calculate_probabilities=True)
    topics, probs = model.fit_transform(docs)
    return model, topics, probs


def get_topic_info(model: BERTopic) -> pd.DataFrame:
    """토픽별 대표 키워드 및 문서 수 반환."""
    return model.get_topic_info()
