"""
직무군 분류 모델 (주 모델 — Validity Guardrail #3)
라벨: Frontend / Backend / Fullstack / Data·AI / DevOps·Cloud / Security / Other IT
입력: 채용제목 + 직무내용 + 자격요건 + 우대사항 + skills

모델 후보:
  - TF-IDF + Logistic Regression
  - TF-IDF + SVM
  - TF-IDF + Random Forest
  - Sentence-BERT 임베딩 + Logistic Regression
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

JOB_LABELS = ["Frontend", "Backend", "Fullstack", "Data·AI", "DevOps·Cloud", "Security", "Other IT"]


def build_tfidf_lr_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression 파이프라인."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def build_tfidf_svm_pipeline() -> Pipeline:
    """TF-IDF + LinearSVC 파이프라인."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf",   LinearSVC(class_weight="balanced")),
    ])


def train_and_evaluate(X_train, y_train, X_test, y_test, pipeline: Pipeline) -> dict:
    """학습 + 평가 결과(Accuracy, F1) 반환."""
    # TODO: 구현
    raise NotImplementedError
