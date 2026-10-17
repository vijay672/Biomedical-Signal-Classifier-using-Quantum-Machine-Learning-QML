"""Binary Q1 metrics with label 1 as the abnormal-class proxy."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if len(y_true) != len(predictions):
        raise ValueError("prediction count must match label count")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_label_1": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall_label_1": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_label_1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "predictions": np.asarray(predictions, dtype=int).tolist(),
        "sample_count": int(len(y_true)),
    }
