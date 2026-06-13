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

# role(원천 직무군 컬럼, 12종) → JOB_LABELS(7종) 매핑.
# 분류(06)와 UMAP 시각화(05)가 공유하는 단일 정의 — 중복 정의 금지.
# 명시 라벨이 없는 세부 직무(embedded/qa/mobile/game/other)는 Other IT 로 통합.
ROLE_TO_LABEL = {
    "frontend": "Frontend",
    "backend":  "Backend",
    "fullstack": "Fullstack",
    "ai_ml":    "Data·AI",
    "data":     "Data·AI",
    "devops":   "DevOps·Cloud",
    "security": "Security",
    "other":    "Other IT",
    "embedded": "Other IT",
    "qa":       "Other IT",
    "mobile":   "Other IT",
    "game":     "Other IT",
}


def map_roles_to_labels(roles, default: str = "Other IT") -> list[str]:
    """role 값 시퀀스를 JOB_LABELS(7종)로 매핑."""
    return [ROLE_TO_LABEL.get(r, default) for r in roles]


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
    """학습 + 평가 결과(Accuracy, weighted-F1, classification_report, confusion_matrix) 반환.

    confusion_matrix / classification_report 의 라벨 순서는 JOB_LABELS 로 고정해
    혼동행렬 축(xticklabels=JOB_LABELS)과 일치시킨다.
    """
    from src.modeling.evaluation import evaluate

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    res = evaluate(y_test, y_pred, labels=JOB_LABELS)
    res["model"] = pipeline
    return res
