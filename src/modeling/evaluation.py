"""모델 평가 유틸 — Accuracy, Precision, Recall, F1, Confusion Matrix."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def evaluate(y_true, y_pred, labels=None) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report":    classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "cm":        confusion_matrix(y_true, y_pred, labels=labels),
    }
