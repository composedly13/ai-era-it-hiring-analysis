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
    ROLE_TO_LABEL,  # role(12종)→JOB_LABELS(7종) 매핑 — 05 UMAP 과 공유하는 단일 정의
)
from src.visualization._fonts import setup_korean_font

KR_PATH  = "data/processed/domestic/kr_jobs_clean.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"
MDL_DIR       = "outputs/models"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

# ★ Validity Guardrail #3 — 라벨 누수 차단:
#   라벨 원천인 'role' 과, 직무 카테고리를 그대로 담은 'site_categories' 는 입력 피처에서 제외한다.
#   분류 입력은 공고 제목 + 본문 + 스킬 텍스트만 사용.
FEATURE_COLS = ["title", "body", "skills"]


def _join_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    text = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        text = text + " " + df[c].fillna("").astype(str)
    return text.tolist()


# ---------------------------------------------------------------------------
# Step 1. 라벨 생성
# ---------------------------------------------------------------------------
def step1_build_dataset() -> tuple[list[str], list[str]]:
    """국내공고(domestic) 기준 분류 데이터셋 생성 — 본 논문 본문(4-4-1)에 싣는 결과.

    role(원천 라벨) → JOB_LABELS 매핑, 입력 피처는 title+body+skills (role/site_categories 제외).
    """
    df = pd.read_csv(KR_PATH)
    df = df[df["role"].isin(ROLE_TO_LABEL)].copy()
    labels = df["role"].map(ROLE_TO_LABEL).tolist()
    texts = _join_features(df, FEATURE_COLS)
    print(f"[Step1] 국내공고 분류 데이터셋: {len(texts)}건 / 라벨 {len(set(labels))}종")
    print(f"        입력 피처(누수 차단 후): {FEATURE_COLS}  (제외: role, site_categories)")
    return texts, labels


# ---------------------------------------------------------------------------
# Step 1'. [코드 보존 — 본 논문 본문 분석 미사용] 글로벌(LinkedIn) 분류 경로
#   글로벌 비교는 논문 본문에서 제외(확장분). 코드·로직은 보존한다.
# ---------------------------------------------------------------------------
def step1_build_dataset_linkedin() -> tuple[list[str], list[str]]:
    df = pd.read_csv(LINKEDIN_PATH)
    df = df[df["role"].isin(ROLE_TO_LABEL)].copy()
    labels = df["role"].map(ROLE_TO_LABEL).tolist()
    texts = _join_features(df, ["title", "description", "skills"])
    print(f"[Step1-LinkedIn/보존] {len(texts)}건 (본문 미사용 확장분)")
    return texts, labels


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
    setup_korean_font()
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=JOB_LABELS, yticklabels=JOB_LABELS,
                cmap="Blues")
    plt.title("직무군 분류 Confusion Matrix", fontsize=14)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Step2] 최고 성능 모델: {best_name} "
          f"(Accuracy={results[best_name]['accuracy']:.4f}, "
          f"weighted-F1={results[best_name]['f1']:.4f})")
    print("[Step2] Confusion Matrix 저장:", f"{FIG_DIR}/confusion_matrix.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 본 논문 본문(4-4-1)은 '국내공고 기준' 결과만 사용·보고한다.
    # LinkedIn(글로벌) 분류 경로는 step1_build_dataset_linkedin() 에 코드로 보존되어 있으나
    # 본문 분석에는 사용하지 않는다(확장분).
    print("[06] 직무군 분류 — 국내공고 기준(본문 4-4-1 / repo legacy RQ6).")
    print("     ※ LinkedIn 경로는 코드에 보존되어 있으나 본 논문 본문 분석에는 사용하지 않음.")
    texts, labels = step1_build_dataset()
    step2_train_evaluate(texts, labels)
