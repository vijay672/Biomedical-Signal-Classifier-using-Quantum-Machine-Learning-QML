"""Bounded, validation-led comparisons of three four-qubit VQC variants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.datasets import PreparedDataset, make_synthetic_benchmark
from src.processor_benchmark import BASIS_GATES, PROCESSOR_FILES, ProcessorSpec
from src.vqc import (
    DEFAULT_MAXITER,
    DEFAULT_PARAMETERS_PATH,
    NUM_QUBITS,
    VQCConfig,
    train_vqc_components,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_PATH = ROOT / "results" / "vqc_controlled_experiments.json"
DEFAULT_SELECTED_PATH = ROOT / "results" / "vqc_selected_parameters.json"
VALIDATION_FRACTION = 0.2
VALIDATION_SEED = 43
MODEL_SEED = 42
TRANSPILER_SEED = 42
OPTIMIZATION_LEVEL = 1


@dataclass(frozen=True)
class Variant:
    name: str
    entanglement: str
    readout: str
    rationale: str


VARIANTS = (
    Variant(
        "original_ring_parity",
        "ring",
        "parity",
        "Original architecture, retrained on the inner training rows for a fair comparison.",
    ),
    Variant(
        "line_parity",
        "line",
        "parity",
        "Remove the ring-closing CX in each layer to reduce line-topology routing.",
    ),
    Variant(
        "ring_first_qubit",
        "ring",
        "first_qubit",
        "Keep the ansatz and test whether a local readout improves positive-class recall.",
    ),
)


@dataclass(frozen=True)
class InnerSplit:
    X_fit: np.ndarray
    X_validation: np.ndarray
    y_fit: np.ndarray
    y_validation: np.ndarray
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    scaler: MinMaxScaler


def make_inner_split(dataset: PreparedDataset) -> InnerSplit:
    """Split only outer training rows; fit the scaler on inner training rows."""

    indices = np.arange(len(dataset.y_train))
    fit_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION,
        random_state=VALIDATION_SEED,
        stratify=dataset.y_train,
    )
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)), clip=True)
    X_fit = scaler.fit_transform(dataset.raw_X_train[fit_indices])
    X_validation = scaler.transform(dataset.raw_X_train[validation_indices])
    return InnerSplit(
        X_fit=X_fit,
        X_validation=X_validation,
        y_fit=dataset.y_train[fit_indices],
        y_validation=dataset.y_train[validation_indices],
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        scaler=scaler,
    )


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Score label 1 as the positive or abnormal proxy class."""

    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    return {
        "recall_label_1": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "precision_label_1": float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "f1_label_1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions.tolist(),
        "sample_count": len(y_true),
    }


def compilation_metrics(circuit: Any) -> dict[str, Any]:
    """Measure circuit and custom-map costs with one fixed transpiler setup."""

    def stats(compiled: Any) -> dict[str, int]:
        return {"cx_count": int(compiled.count_ops().get("cx", 0)), "depth": int(compiled.depth())}

    raw = stats(circuit)
    common = dict(
        basis_gates=list(BASIS_GATES),
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=TRANSPILER_SEED,
    )
    unconstrained = stats(transpile(circuit, **common))
    processors: dict[str, Any] = {}
    for path in PROCESSOR_FILES:
        spec = ProcessorSpec.from_json(path)
        routed = stats(
            transpile(circuit, coupling_map=CouplingMap(list(spec.coupling_edges)), **common)
        )
        processors[spec.name] = {
            "coupling_map_source": spec.coupling_map_source,
            "cx_count": routed["cx_count"],
            "depth": routed["depth"],
            "routing_overhead_cx": routed["cx_count"] - unconstrained["cx_count"],
            "routing_overhead_depth": routed["depth"] - unconstrained["depth"],
        }
    return {"raw": raw, "unconstrained": unconstrained, "processors": processors}


def select_variant(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by validation recall, then precision, F1, routing, and fixed order."""

    if not 1 <= len(candidates) <= 3:
        raise ValueError("selection requires one to three candidates")

    def rank(item: tuple[int, dict[str, Any]]) -> tuple[float, float, float, int, int]:
        index, candidate = item
        metrics = candidate["validation"]
        processors = candidate["compilation"]["processors"]
        routing_cx = sum(value["routing_overhead_cx"] for value in processors.values())
        return (
            metrics["recall_label_1"],
            metrics["precision_label_1"],
            metrics["f1_label_1"],
            -routing_cx,
            -index,
        )

    return max(enumerate(candidates), key=rank)[1]


def _sha256_if_present(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run_controlled_experiments(
    experiment_path: Path | str = DEFAULT_EXPERIMENT_PATH,
    selected_path: Path | str = DEFAULT_SELECTED_PATH,
) -> dict[str, Any]:
    """Compare three candidates on validation, then evaluate the selected refit once."""

    baseline_files = {
        "original_vqc_parameters": DEFAULT_PARAMETERS_PATH,
        "logistic_regression_results": ROOT / "results" / "logistic_regression_baseline.json",
    }
    baseline_before = {name: _sha256_if_present(path) for name, path in baseline_files.items()}
    dataset = make_synthetic_benchmark()
    if dataset.raw_X_train.shape[1] != NUM_QUBITS:
        raise ValueError("controlled experiment requires four input features")
    inner = make_inner_split(dataset)

    candidates: list[dict[str, Any]] = []
    for variant in VARIANTS:
        config = VQCConfig(
            random_state=MODEL_SEED,
            maxiter=DEFAULT_MAXITER,
            shots=None,
            entanglement=variant.entanglement,
            readout=variant.readout,
        )
        components = train_vqc_components(inner.X_fit, inner.y_fit, config)
        probabilities = np.asarray(
            components.classifier.predict_proba(inner.X_validation), dtype=np.float64
        )
        if probabilities.shape != (len(inner.y_validation), 2):
            raise RuntimeError("unexpected validation probability shape")
        candidates.append(
            {
                "variant": asdict(variant),
                "config": asdict(config),
                "validation": classification_metrics(inner.y_validation, probabilities),
                "compilation": compilation_metrics(components.circuit),
                "training": {
                    "fit_samples": len(inner.y_fit),
                    "optimizer_evaluations": int(components.classifier.fit_result.nfev),
                    "final_objective": float(components.classifier.fit_result.fun),
                    "trained_parameters": [
                        {"name": str(parameter), "value_radians": float(value)}
                        for parameter, value in zip(
                            components.weight_parameters,
                            components.classifier.weights,
                            strict=True,
                        )
                    ],
                },
            }
        )

    selected = select_variant(candidates)
    selected_variant = selected["variant"]
    selected_config = VQCConfig(
        random_state=MODEL_SEED,
        maxiter=DEFAULT_MAXITER,
        shots=None,
        entanglement=selected_variant["entanglement"],
        readout=selected_variant["readout"],
    )
    # Selection is complete before any held-out features or labels are used.
    final_components = train_vqc_components(dataset.X_train, dataset.y_train, selected_config)
    final_probabilities = np.asarray(
        final_components.classifier.predict_proba(dataset.X_test), dtype=np.float64
    )
    if final_probabilities.shape != (len(dataset.y_test), 2):
        raise RuntimeError("unexpected held-out probability shape")
    final_metrics = classification_metrics(dataset.y_test, final_probabilities)
    selected_payload = {
        "schema_version": 1,
        "selection_basis": "inner validation only",
        "selected_variant": selected_variant,
        "architecture": {
            "num_qubits": NUM_QUBITS,
            "entanglement": selected_config.entanglement,
            "readout": selected_config.readout,
        },
        "trained_parameters": [
            {"name": str(parameter), "value_radians": float(value)}
            for parameter, value in zip(
                final_components.weight_parameters,
                final_components.classifier.weights,
                strict=True,
            )
        ],
        "training": {
            "fit_samples": len(dataset.y_train),
            "optimizer": "COBYLA",
            "maxiter": selected_config.maxiter,
            "optimizer_evaluations": int(final_components.classifier.fit_result.nfev),
            "final_objective": float(final_components.classifier.fit_result.fun),
            "random_state": selected_config.random_state,
            "simulator": "QMLSampler exact statevector",
        },
        "dataset": {
            "source": dataset.source,
            "generation_random_state": dataset.generation_random_state,
            "outer_split_random_state": dataset.random_state,
            "outer_training_samples": len(dataset.y_train),
            "outer_test_samples": len(dataset.y_test),
            "preprocessing": "outer training fitted MinMaxScaler to [0, pi]",
        },
        "held_out_metrics_after_selection": final_metrics,
    }
    selected_destination = Path(selected_path)
    selected_destination.parent.mkdir(parents=True, exist_ok=True)
    selected_destination.write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    baseline_after = {name: _sha256_if_present(path) for name, path in baseline_files.items()}
    if baseline_before != baseline_after:
        raise RuntimeError("original baseline artifacts changed during the experiments")
    try:
        selected_file_label = str(selected_destination.resolve().relative_to(ROOT))
    except ValueError:
        selected_file_label = str(selected_destination.resolve())
    report = {
        "schema_version": 1,
        "problem": "binary classification of the Q1 four-feature synthetic benchmark",
        "protocol": {
            "candidate_count": len(VARIANTS),
            "outer_training_samples": len(dataset.y_train),
            "inner_fit_samples": len(inner.y_fit),
            "inner_validation_samples": len(inner.y_validation),
            "held_out_test_samples": len(dataset.y_test),
            "inner_validation_fraction": VALIDATION_FRACTION,
            "inner_validation_seed": VALIDATION_SEED,
            "fit_indices_within_outer_training": inner.fit_indices.tolist(),
            "validation_indices_within_outer_training": inner.validation_indices.tolist(),
            "preprocessing": "MinMaxScaler fitted only on inner fit rows, then applied to validation",
            "inner_scaler_data_min": inner.scaler.data_min_.tolist(),
            "inner_scaler_data_max": inner.scaler.data_max_.tolist(),
            "model_seed": MODEL_SEED,
            "simulator": "QMLSampler exact statevector, shots=None",
            "optimizer": "COBYLA",
            "optimizer_maxiter_per_fit": DEFAULT_MAXITER,
            "basis_gates": list(BASIS_GATES),
            "transpiler_optimization_level": OPTIMIZATION_LEVEL,
            "transpiler_seed": TRANSPILER_SEED,
            "routing_overhead_definition": "CX/depth delta from unconstrained compile",
            "selection_rule": (
                "maximize validation label-1 recall; break ties by precision, F1, "
                "lower sum of A/B routing CX deltas, then listed candidate order"
            ),
            "held_out_use": "evaluated once after selection and full outer-training refit",
        },
        "candidates": candidates,
        "selected_variant_name": selected_variant["name"],
        "selected_validation": selected["validation"],
        "selected_parameter_file": selected_file_label,
        "selected_parameter_file_sha256": hashlib.sha256(selected_destination.read_bytes()).hexdigest(),
        "final_refit_and_held_out": {
            "fit_samples": len(dataset.y_train),
            "optimizer_evaluations": int(final_components.classifier.fit_result.nfev),
            "metrics": final_metrics,
        },
        "baseline_artifact_sha256_unchanged": baseline_after,
        "environment": {
            "qiskit": version("qiskit"),
            "qiskit_machine_learning": version("qiskit-machine-learning"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    experiment_destination = Path(experiment_path)
    experiment_destination.parent.mkdir(parents=True, exist_ok=True)
    experiment_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run_controlled_experiments()
    for candidate in result["candidates"]:
        metrics = candidate["validation"]
        print(
            f"{candidate['variant']['name']}: validation recall={metrics['recall_label_1']:.3f}, "
            f"precision={metrics['precision_label_1']:.3f}, F1={metrics['f1_label_1']:.3f}"
        )
    print(f"Selected {result['selected_variant_name']} by validation only")
    print(f"Saved {DEFAULT_EXPERIMENT_PATH} and {DEFAULT_SELECTED_PATH}")
