"""One bounded, separately labeled VQC validation and recall calibration run."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.artifacts import architecture_identity
from src.datasets import make_provisional_q1_benchmark
from src.metrics import classification_metrics
from src.problem import dataset_identity
from src.q1_submission import CONFIG, ROOT
from src.vqc import train_vqc_components


DEFAULT_OUTPUT_DIR = ROOT / "results" / "tables"
RESULT_NAME = "q1_recall_calibration.json"
MODEL_NAME = "q1_recall_calibration_model.json"
INNER_SPLIT_SEED = 43
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
MIN_PRECISION = 0.55
MIN_F1 = 0.60


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode()
        + array.tobytes()
    ).hexdigest()


def choose_threshold(
    y_validation: np.ndarray,
    positive_scores: np.ndarray,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    min_precision: float = MIN_PRECISION,
    min_f1: float = MIN_F1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select by validation recall subject to fixed precision and F1 floors."""

    if (y_validation.ndim != 1 or positive_scores.ndim != 1
            or y_validation.shape != positive_scores.shape
            or not np.isfinite(positive_scores).all()
            or not np.isin(y_validation, (0, 1)).all()):
        raise ValueError("validation labels and scores must be finite vectors of equal length")
    grid = []
    for threshold in thresholds:
        metrics = classification_metrics(y_validation, (positive_scores > threshold).astype(int))
        grid.append({
            "threshold": threshold,
            "eligible": (
                metrics["precision_label_1"] >= min_precision
                and metrics["f1_label_1"] >= min_f1
            ),
            "metrics": metrics,
        })
    eligible = [row for row in grid if row["eligible"]]
    if eligible:
        chosen = max(eligible, key=lambda row: (
            row["metrics"]["recall_label_1"],
            row["metrics"]["f1_label_1"],
            row["metrics"]["precision_label_1"],
            row["threshold"],
        ))
        status = "selected on inner validation only"
    else:
        chosen = next(row for row in grid if row["threshold"] == 0.5)
        status = "no candidate met both constraints; fallback to 0.5"
    return {"threshold": chosen["threshold"], "status": status,
            "validation_metrics": chosen["metrics"]}, grid


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Train once on 128 fit rows, select on 32 validation rows, test once."""

    result_path = output_dir / RESULT_NAME
    model_path = output_dir / MODEL_NAME
    if result_path.exists() or model_path.exists():
        if not result_path.is_file() or not model_path.is_file():
            raise RuntimeError("partial recall experiment exists; inspect it before rerunning")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if (result.get("model_file_sha256") != model_hash or
                result.get("outer_dataset_sha256") != dataset_identity(
                    make_provisional_q1_benchmark())["dataset_sha256"]):
            raise ValueError("existing recall experiment has a model or dataset mismatch")
        return result

    outer = make_provisional_q1_benchmark()
    outer_identity = dataset_identity(outer)
    indices = np.arange(len(outer.y_train))
    fit_idx, validation_idx = train_test_split(
        indices, test_size=0.2, random_state=INNER_SPLIT_SEED,
        stratify=outer.y_train,
    )
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)), clip=True)
    X_fit = scaler.fit_transform(outer.raw_X_train[fit_idx])
    X_validation = scaler.transform(outer.raw_X_train[validation_idx])
    if scaler.n_samples_seen_ != len(fit_idx):
        raise RuntimeError("inner scaler was not fitted exclusively on fit rows")
    convergence: list[dict[str, float | int]] = []

    def callback(_weights: np.ndarray, objective: float) -> None:
        convergence.append({"evaluation": len(convergence) + 1, "objective": float(objective)})

    started = perf_counter()
    components = train_vqc_components(
        X_fit, outer.y_train[fit_idx], CONFIG, callback=callback,
    )
    elapsed = perf_counter() - started
    weights = np.asarray(components.classifier.weights, dtype=float)
    fit_result = components.classifier.fit_result
    if len(convergence) != int(fit_result.nfev):
        raise RuntimeError("training callback did not record every optimizer evaluation")
    validation_probability = np.asarray(components.qnn.forward(X_validation, weights), dtype=float)
    if validation_probability.shape != (len(validation_idx), 2):
        raise RuntimeError("inner validation QNN output shape is invalid")
    validation_scores = validation_probability[:, 1]
    selection, grid = choose_threshold(outer.y_train[validation_idx], validation_scores)

    # The outer test is transformed and scored only after the validation decision.
    X_test = scaler.transform(outer.raw_X_test)
    test_probability = np.asarray(components.qnn.forward(X_test, weights), dtype=float)
    if test_probability.shape != (len(outer.y_test), 2):
        raise RuntimeError("outer test QNN output shape is invalid")
    test_scores = test_probability[:, 1]
    test_predictions = (test_scores > selection["threshold"]).astype(int)
    model = {
        "schema_version": 1,
        "experiment": "separate inner-fit VQC; official full-training model unchanged",
        "architecture": architecture_identity(CONFIG),
        "outer_dataset_sha256": outer_identity["dataset_sha256"],
        "fit_indices_in_outer_training": fit_idx.tolist(),
        "validation_indices_in_outer_training": validation_idx.tolist(),
        "fit_raw_array_sha256": _hash_array(outer.raw_X_train[fit_idx]),
        "validation_raw_array_sha256": _hash_array(outer.raw_X_train[validation_idx]),
        "outer_test_raw_array_sha256": _hash_array(outer.raw_X_test),
        "inner_scaler": {
            "type": "MinMaxScaler",
            "feature_range": list(scaler.feature_range),
            "clip": bool(scaler.clip),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "scale": scaler.scale_.tolist(),
            "min": scaler.min_.tolist(),
        },
        "trained_parameters": [
            {"name": str(parameter), "value_radians": float(value)}
            for parameter, value in zip(components.weight_parameters, weights, strict=True)
        ],
        "training": {
            "optimizer": "COBYLA", "maxiter": CONFIG.maxiter,
            "random_state": CONFIG.random_state,
            "optimizer_evaluations": int(fit_result.nfev),
            "final_objective": float(fit_result.fun),
            "elapsed_seconds_observed": elapsed,
            "convergence": convergence,
        },
        "environment": {
            "qiskit": version("qiskit"),
            "qiskit_machine_learning": version("qiskit-machine-learning"),
            "scikit_learn": version("scikit-learn"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
    }
    _write(model_path, model)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    result = {
        "schema_version": 1,
        "status": "separate one-run threshold experiment; not the official full-training baseline",
        "outer_dataset_sha256": outer_identity["dataset_sha256"],
        "outer_test_label_counts_0_1": np.bincount(outer.y_test, minlength=2).tolist(),
        "fit_label_counts_0_1": np.bincount(outer.y_train[fit_idx], minlength=2).tolist(),
        "validation_label_counts_0_1": np.bincount(outer.y_train[validation_idx], minlength=2).tolist(),
        "split": {
            "method": "stratified 128/32 within the original 160 training rows",
            "seed": INNER_SPLIT_SEED,
            "fit_rows": len(fit_idx), "validation_rows": len(validation_idx),
            "test_rows": len(outer.y_test),
            "inner_scaler_fit_rows": int(scaler.n_samples_seen_),
        },
        "threshold_search": {
            "predeclared_candidates": list(THRESHOLDS),
            "minimum_validation_precision": MIN_PRECISION,
            "minimum_validation_f1": MIN_F1,
            "selection_priority": "highest recall, then F1, precision, higher threshold",
            "selection": selection,
            "grid": grid,
            "validation_positive_scores": validation_scores.tolist(),
            "validation_labels": outer.y_train[validation_idx].tolist(),
        },
        "held_out_test": {
            "evaluated_once_after_selection": True,
            "selected_threshold": selection["threshold"],
            "positive_scores": test_scores.tolist(),
            "metrics": classification_metrics(outer.y_test, test_predictions),
        },
        "model_file": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "model_file_sha256": model_hash,
    }
    _write(result_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run(args.output_dir)
    print(args.output_dir / RESULT_NAME)
    print("validation choice", output["threshold_search"]["selection"])
    print("held-out metrics", output["held_out_test"]["metrics"])
