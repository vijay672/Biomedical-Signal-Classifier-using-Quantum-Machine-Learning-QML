"""Recalculate paired held-out evidence and descriptive uncertainty from saved outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, norm
from sklearn.model_selection import train_test_split

from src.datasets import make_provisional_q1_benchmark
from src.metrics import classification_metrics
from src.problem import dataset_identity
from src.q1_submission import BASELINE_PATH, BENCHMARK_PATH, IDEAL_PATH, ROOT
from src.recall_calibration import (
    INNER_SPLIT_SEED, MODEL_NAME, RESULT_NAME, _hash_array, choose_threshold,
)


OUTPUT_PATH = ROOT / "results" / "tables" / "q1_statistical_review.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(successes: int, trials: int, confidence: float = .95) -> list[float]:
    """Two-sided Wilson score interval for a binomial proportion."""

    if trials < 1 or not 0 <= successes <= trials or not 0 < confidence < 1:
        raise ValueError("invalid Wilson interval inputs")
    z = float(norm.ppf((1 + confidence) / 2))
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [float(center - half_width), float(center + half_width)]


def _case(y: np.ndarray, stored: dict[str, Any]) -> dict[str, Any]:
    predictions = np.asarray(stored["metrics"]["predictions"], dtype=int)
    if predictions.shape != y.shape or not np.isin(predictions, (0, 1)).all():
        raise ValueError("saved prediction vector is not aligned with the fixed test labels")
    recalculated = classification_metrics(y, predictions)
    if recalculated != stored["metrics"]:
        raise ValueError("saved classification metrics disagree with predictions")
    (tn, fp), (fn, tp) = recalculated["confusion_matrix"]
    return {
        "metrics": {key: value for key, value in recalculated.items() if key != "predictions"},
        "specificity_label_0": tn / (tn + fp),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "positive_prediction_count": int(predictions.sum()),
        "wilson_95_percent": {
            "accuracy": wilson_interval(tp + tn, len(y)),
            "recall_label_1": wilson_interval(tp, tp + fn),
            "specificity_label_0": wilson_interval(tn, tn + fp),
        },
    }


def _paired(y: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    first_correct, second_correct = first == y, second == y
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    positive = y == 1
    first_only_recall = int(np.sum((first[positive] == 1) & (second[positive] == 0)))
    second_only_recall = int(np.sum((first[positive] == 0) & (second[positive] == 1)))

    def exact_p(a: int, b: int) -> float:
        return float(binomtest(min(a, b), a + b, .5).pvalue) if a + b else 1.0

    return {
        "accuracy_discordance_first_only_correct": first_only,
        "accuracy_discordance_second_only_correct": second_only,
        "exact_two_sided_mcnemar_accuracy_p": exact_p(first_only, second_only),
        "recall_discordance_first_only_detects_label_1": first_only_recall,
        "recall_discordance_second_only_detects_label_1": second_only_recall,
        "exact_two_sided_paired_recall_p": exact_p(first_only_recall, second_only_recall),
    }


def run(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    dataset = make_provisional_q1_benchmark()
    identity = dataset_identity(dataset)
    y = dataset.y_test
    baseline, ideal, benchmark = map(_load, (BASELINE_PATH, IDEAL_PATH, BENCHMARK_PATH))
    tables = ROOT / "results" / "tables"
    separate = _load(tables / RESULT_NAME)
    separate_model = _load(tables / MODEL_NAME)
    if (baseline["dataset_identity"] != identity
            or ideal["dataset_sha256"] != identity["dataset_sha256"]
            or benchmark["dataset_identity"] != identity
            or separate["outer_dataset_sha256"] != identity["dataset_sha256"]
            or separate_model["outer_dataset_sha256"] != identity["dataset_sha256"]
            or separate_model["outer_test_raw_array_sha256"] != _hash_array(dataset.raw_X_test)):
        raise ValueError("official and separate models do not share the same outer held-out rows")

    indices = np.arange(len(dataset.y_train))
    fit, validation = train_test_split(
        indices, test_size=.2, random_state=INNER_SPLIT_SEED,
        stratify=dataset.y_train,
    )
    if (separate_model["fit_indices_in_outer_training"] != fit.tolist()
            or separate_model["validation_indices_in_outer_training"] != validation.tolist()
            or separate_model["fit_raw_array_sha256"] != _hash_array(dataset.raw_X_train[fit])
            or separate_model["validation_raw_array_sha256"] != _hash_array(dataset.raw_X_train[validation])):
        raise ValueError("inner fit or validation partition does not match its saved identity")
    scaler = separate_model["inner_scaler"]
    if (scaler["n_samples_seen"] != len(fit)
            or not np.allclose(scaler["data_min"], dataset.raw_X_train[fit].min(axis=0))
            or not np.allclose(scaler["data_max"], dataset.raw_X_train[fit].max(axis=0))):
        raise ValueError("inner scaler contains statistics outside the 128 fit rows")
    search = separate["threshold_search"]
    validation_y = dataset.y_train[validation]
    if search["validation_labels"] != validation_y.tolist():
        raise ValueError("saved validation labels disagree with the independent inner split")
    selected, grid = choose_threshold(
        validation_y, np.asarray(search["validation_positive_scores"], dtype=float),
        thresholds=tuple(search["predeclared_candidates"]),
        min_precision=search["minimum_validation_precision"],
        min_f1=search["minimum_validation_f1"],
    )
    if selected != search["selection"] or grid != search["grid"]:
        raise ValueError("saved threshold was not selected by validation-only rules")
    threshold = selected["threshold"]
    test_scores = np.asarray(separate["held_out_test"]["positive_scores"], dtype=float)
    if (test_scores.shape != y.shape or not np.isfinite(test_scores).all()
            or threshold != separate["held_out_test"]["selected_threshold"]
            or (test_scores > threshold).astype(int).tolist()
            != separate["held_out_test"]["metrics"]["predictions"]):
        raise ValueError("separate held-out predictions disagree with the selected threshold")

    stored_cases = {
        "logistic": baseline,
        "ideal_vqc": ideal,
        "processor_A_ideal": benchmark["processors"][0]["ideal"],
        "processor_A_noisy": benchmark["processors"][0]["noisy"],
        "processor_B_ideal": benchmark["processors"][1]["ideal"],
        "processor_B_noisy": benchmark["processors"][1]["noisy"],
        "separate_validation_model": separate["held_out_test"],
    }
    cases = {name: _case(y, item) for name, item in stored_cases.items()}
    predictions = {name: np.asarray(item["metrics"]["predictions"], dtype=int)
                   for name, item in stored_cases.items()}
    comparisons = {
        f"{first}_vs_{second}": _paired(y, predictions[first], predictions[second])
        for first, second in (
            ("logistic", "ideal_vqc"),
            ("ideal_vqc", "processor_A_noisy"),
            ("ideal_vqc", "processor_B_noisy"),
            ("processor_A_noisy", "processor_B_noisy"),
            ("ideal_vqc", "separate_validation_model"),
        )
    }
    result = {
        "schema_version": 1,
        "scope": "descriptive and exploratory paired analysis of one fixed 40-row synthetic test split",
        "dataset_sha256": identity["dataset_sha256"],
        "outer_test_rows_identical": True,
        "inner_fit_rows": len(fit),
        "inner_validation_rows": len(validation),
        "inner_scaler_fit_rows": scaler["n_samples_seen"],
        "validation_only_threshold_verified": True,
        "selected_threshold": threshold,
        "interval_method": "two-sided 95% Wilson binomial intervals; accuracy n=40, recall and specificity n=20",
        "paired_method": "two-sided exact McNemar/binomial test on discordant correctness; recall restricted to 20 label-1 rows",
        "interpretation_limit": "one fixed synthetic test split, reused for post hoc analysis; p-values exploratory, not clinical or broad-population inference",
        "cases": cases,
        "paired_comparisons": comparisons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    output = run()
    print(OUTPUT_PATH, "verified", len(output["cases"]), "cases and",
          len(output["paired_comparisons"]), "paired contrasts")
