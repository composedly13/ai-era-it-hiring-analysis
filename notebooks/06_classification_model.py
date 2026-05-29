"""
06_classification_model.py
담당: 미정

직무군 분류 모델 (보조 모델, RQ6)
라벨: Frontend / Backend / Fullstack / Data·AI / DevOps·Cloud / Security / Other IT
출력: outputs/models/, outputs/figures/confusion_matrix.png

Validity Guardrail #3 — 순환논리 회피:
  키워드로 라벨링 후 같은 키워드로 학습하면 성능이 무의미하게 높아진다.
  직무군 분류는 국내공고 직종코드 / LinkedIn job title 기반 라벨을 사용한다.
  AI 관련 공고 분류는 보조(키워드 약지도)로만 활용하고 한계를 논문에 명시한다.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.modeling.job_classifier import (
    build_tfidf_lr_pipeline,
    build_tfidf_svm_pipeline,
    train_and_evaluate,
    JOB_LABELS,
)
from src.modeling.evaluation import evaluate

KR_PATH  = "data/processed/domestic/kr_jobs_clean.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"
MDL_DIR       = "outputs/models"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1. 라벨 생성
# ---------------------------------------------------------------------------
def step1_build_dataset() -> tuple[list[str], list[str]]:
    df_w = pd.read_csv(KR_PATH)
    df_l = pd.read_csv(LINKEDIN_PATH)

    # TODO: 국내공고 직종코드 → JOB_LABELS 매핑
    # TODO: LinkedIn job_title → JOB_LABELS 매핑
    # texts = ...
    # labels = ...
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2. 학습 / 평가
# ---------------------------------------------------------------------------
def step2_train_evaluate(texts: list[str], labels: list[str]) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    results = {}
    for name, pipeline in [
        ("TF-IDF + LR",  build_tfidf_lr_pipeline()),
        ("TF-IDF + SVM", build_tfidf_svm_pipeline()),
    ]:
        res = train_and_evaluate(X_train, y_train, X_test, y_test, pipeline)
        results[name] = res
        print(f"\n[{name}] Accuracy: {res['accuracy']:.4f} / F1: {res['f1']:.4f}")
        print(res["report"])

    # Confusion Matrix (최고 성능 모델)
    best_name = max(results, key=lambda k: results[k]["f1"])
    cm = results[best_name]["cm"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=JOB_LABELS, yticklabels=JOB_LABELS,
                cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}", fontsize=13)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Step2] Confusion Matrix 저장:", f"{FIG_DIR}/confusion_matrix.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    texts, labels = step1_build_dataset()
    step2_train_evaluate(texts, labels)
