"""Reproducible classical baseline using the shared Q1 data split."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import sklearn

from src.datasets import PreparedDataset, make_synthetic_benchmark


DEFAULT_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "logistic_regression_baseline.json"


@dataclass(frozen=True)
class BaselineResult:
    """Held-out test metrics plus the settings needed to reproduce them."""

    accuracy: float
    abnormal_recall: float
    precision: float
    f1: float
    confusion_matrix: list[list[int]]
    confusion_matrix_labels: list[int]
    test_samples: int
    positive_label: int
    model: dict[str, Any]
    data: dict[str, Any]
    sklearn_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_logistic_regression(
    dataset: PreparedDataset,
    *,
    random_state: int = 42,
) -> BaselineResult:
    """Fit on the supplied training partition and score its held-out test set.

    The function consumes the already-prepared arrays and does not split or
    preprocess them again. Class label 1 is treated as the abnormal class.
    """

    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(dataset.X_train, dataset.y_train)
    predictions = model.predict(dataset.X_test)
    labels = [0, 1]

    return BaselineResult(
        accuracy=float(accuracy_score(dataset.y_test, predictions)),
        abnormal_recall=float(
            recall_score(dataset.y_test, predictions, pos_label=1, zero_division=0)
        ),
        precision=float(
            precision_score(dataset.y_test, predictions, pos_label=1, zero_division=0)
        ),
        f1=float(f1_score(dataset.y_test, predictions, pos_label=1, zero_division=0)),
        confusion_matrix=confusion_matrix(
            dataset.y_test, predictions, labels=labels
        ).astype(int).tolist(),
        confusion_matrix_labels=labels,
        test_samples=int(dataset.y_test.shape[0]),
        positive_label=1,
        model={
            "name": "LogisticRegression",
            "random_state": random_state,
            "max_iter": 1000,
        },
        data={
            "source": dataset.source,
            "n_samples": int(dataset.raw_X_train.shape[0] + dataset.raw_X_test.shape[0]),
            "n_features": int(dataset.raw_X_train.shape[1]),
            "train_samples": int(dataset.y_train.shape[0]),
            "test_samples": int(dataset.y_test.shape[0]),
            "generation_random_state": dataset.generation_random_state,
            "split": (
                "stratified train/test split "
                f"({1 - dataset.test_size:.0%}/{dataset.test_size:.0%}), "
                f"random_state={dataset.random_state}"
            ),
            "preprocessing": "MinMaxScaler to [0, pi], fit on training rows only",
            "feature_names": list(dataset.feature_names),
        },
        sklearn_version=sklearn.__version__,
    )


def save_baseline_result(
    result: BaselineResult,
    path: Path | str = DEFAULT_RESULT_PATH,
) -> Path:
    """Write baseline metrics and reproduction metadata as formatted JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def run_synthetic_baseline(
    output_path: Path | str = DEFAULT_RESULT_PATH,
) -> BaselineResult:
    """Prepare the standard synthetic split, evaluate, and save its metrics."""

    dataset = make_synthetic_benchmark()
    result = evaluate_logistic_regression(dataset)
    save_baseline_result(result, output_path)
    return result


if __name__ == "__main__":
    result = run_synthetic_baseline()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
