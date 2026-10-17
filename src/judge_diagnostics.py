"""Replay saved Q1 evidence and compare deterministic layout choices."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from src.artifacts import load_verified_model
from src.datasets import make_provisional_q1_benchmark
from src.metrics import classification_metrics
from src.problem import dataset_identity
from src.processor_benchmark import make_noise_model
from src.processors import ProcessorSpec
from src.q1_submission import (
    BASELINE_PATH, BENCHMARK_PATH, CONFIG, IDEAL_PATH, MANIFEST_PATH,
    MODEL_PATH, PRESCRIBED_NOISE, PROCESSOR_PATHS, ROOT, SHOTS,
)
from src.routing import COMPILER_SETTINGS, circuit_cost, compile_processor, compile_unconstrained, layout_record
from src.vqc import build_vqc_circuit, build_vqc_components


TABLES = ROOT / "results" / "tables"
AUDIT_PATH = TABLES / "q1_judge_audit.json"
ABLATION_PATH = TABLES / "q1_layout_ablation.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _metric_record(y: np.ndarray, record: dict[str, Any]) -> dict[str, Any]:
    stored = record["metrics"]
    predictions = np.asarray(stored["predictions"], dtype=np.int64)
    recomputed = classification_metrics(y, predictions)
    for key, value in recomputed.items():
        if stored[key] != value:
            raise ValueError(f"saved metric mismatch: {key}")
    result: dict[str, Any] = {
        "metrics": {key: value for key, value in recomputed.items() if key != "predictions"},
        "positive_prediction_count": int(predictions.sum()),
    }
    if "positive_probabilities" in record:
        scores = np.asarray(record["positive_probabilities"], dtype=float)
        if scores.shape != y.shape or not np.isfinite(scores).all():
            raise ValueError("saved positive-class score shape or values are invalid")
        if not np.array_equal(predictions, (scores > 0.5).astype(np.int64)):
            raise ValueError("saved predictions disagree with the 0.5 score threshold")
        result["score_diagnostics"] = {
            "mean_by_true_class": {
                str(label): float(scores[y == label].mean()) for label in (0, 1)
            },
            "label_1_quantiles": np.quantile(scores[y == 1], [0, .25, .5, .75, 1]).tolist(),
            "label_0_quantiles": np.quantile(scores[y == 0], [0, .25, .5, .75, 1]).tolist(),
        }
    return result


def audit_saved_results() -> dict[str, Any]:
    dataset = make_provisional_q1_benchmark()
    identity = dataset_identity(dataset)
    baseline = _load(BASELINE_PATH)
    ideal = _load(IDEAL_PATH)
    benchmark = _load(BENCHMARK_PATH)
    manifest = _load(MANIFEST_PATH)
    model = _load(MODEL_PATH)
    weights, model_sha = load_verified_model(MODEL_PATH, dataset, CONFIG)
    if baseline["dataset_identity"] != identity or benchmark["dataset_identity"] != identity:
        raise ValueError("classical and processor records do not share the corrected dataset")
    if ideal["dataset_sha256"] != identity["dataset_sha256"]:
        raise ValueError("ideal VQC uses a different dataset")
    if benchmark["model_file_sha256"] != model_sha or manifest["dataset_identity"] != identity:
        raise ValueError("benchmark or manifest model/data identity mismatch")
    if dataset.scaler.n_samples_seen_ != len(dataset.y_train):
        raise ValueError("scaler was not fitted to training rows only")
    np.testing.assert_allclose(dataset.scaler.data_min_, dataset.raw_X_train.min(axis=0))
    np.testing.assert_allclose(dataset.scaler.data_max_, dataset.raw_X_train.max(axis=0))
    np.testing.assert_allclose(dataset.X_test, dataset.scaler.transform(dataset.raw_X_test))
    settings = benchmark["shared_settings"]
    if settings != {
        "basis_gates": list(COMPILER_SETTINGS["basis_gates"]),
        "optimization_level": 1,
        "layout_method": "sabre",
        "routing_method": "sabre",
        "seed_transpiler": 42,
        "seed_simulator": 42,
        "shots_per_circuit": SHOTS,
        "noise_rates": asdict(PRESCRIBED_NOISE),
        "noise_model": "uniform depolarizing on rz/sx/x and cx; symmetric readout bit-flip",
    }:
        raise ValueError("saved processor settings differ from the prescribed baseline")
    noise = make_noise_model(PRESCRIBED_NOISE)
    if set(noise.noise_instructions) != {"rz", "sx", "x", "cx", "measure"}:
        raise ValueError("noise model targets the wrong operations")
    readout = next(error for error in noise.to_dict()["errors"] if error["type"] == "roerror")
    if readout["probabilities"] != [[.99, .01], [.01, .99]]:
        raise ValueError("readout channel is not symmetric with p=0.01")

    logical, _, _ = build_vqc_circuit(CONFIG.entanglement)
    reference = circuit_cost(compile_unconstrained(logical))
    if reference != benchmark["unconstrained_circuit_cost"]:
        raise ValueError("unconstrained reference cost changed")
    cases = {
        "logistic_regression": _metric_record(dataset.y_test, baseline),
        "ideal_vqc_exact": _metric_record(dataset.y_test, ideal),
    }
    qnn = build_vqc_components(CONFIG).qnn
    exact_scores = np.asarray(qnn.forward(dataset.X_test, weights), dtype=float)[:, 1]
    np.testing.assert_allclose(exact_scores, ideal["positive_probabilities"], rtol=0, atol=1e-12)
    final_rz_sensitivity = {}
    for index in (9, 11, 13, 15):
        perturbed = weights.copy()
        perturbed[index] += 0.37
        changed = np.asarray(qnn.forward(dataset.X_test, perturbed), dtype=float)[:, 1]
        final_rz_sensitivity[f"w[{index}]"] = float(np.max(np.abs(changed - exact_scores)))
    if len(benchmark["processors"]) != 2:
        raise ValueError("expected exactly two processor records")
    topology_records = []
    for path, saved in zip(PROCESSOR_PATHS, benchmark["processors"], strict=True):
        spec = ProcessorSpec.from_json(path)
        compiled = compile_processor(logical, spec)
        cost = circuit_cost(compiled)
        if saved["name"] != spec.name or saved["circuit_cost"] != cost:
            raise ValueError(f"{spec.name} circuit costs changed")
        if saved["layout"] != layout_record(logical, compiled):
            raise ValueError(f"{spec.name} layout changed")
        if saved["coupling_edges"] != [list(edge) for edge in spec.coupling_edges]:
            raise ValueError(f"{spec.name} coupling graph changed")
        allowed = set(spec.coupling_edges)
        for instruction in compiled.data:
            if instruction.operation.name == "cx":
                edge = tuple(compiled.find_bit(qubit).index for qubit in instruction.qubits)
                if edge not in allowed:
                    raise ValueError(f"{spec.name} has a CX outside its coupling map")
        reference_delta = {
            "gate_only_depth_delta": cost["gate_only_depth"] - reference["gate_only_depth"],
            "full_measured_depth_delta": cost["full_measured_depth"] - reference["full_measured_depth"],
            "cx_delta": cost["cx_count"] - reference["cx_count"],
        }
        for key, value in reference_delta.items():
            if saved["routing_overhead"][key] != value:
                raise ValueError(f"{spec.name} routing overhead changed: {key}")
        cases[f"{spec.name} ideal sampled"] = _metric_record(dataset.y_test, saved["ideal"])
        cases[f"{spec.name} noisy sampled"] = _metric_record(dataset.y_test, saved["noisy"])
        topology_records.append({
            "name": spec.name,
            "num_qubits": spec.num_qubits,
            "coupling_map_source": spec.coupling_map_source,
            "undirected_edge_count": len({tuple(sorted(edge)) for edge in spec.coupling_edges}),
            "logical_qubit_utilization_of_budget": logical.num_qubits / spec.num_qubits,
            "cost": cost,
            "routing_overhead": reference_delta,
            "ideal_score_difference_from_exact": {
                "mean_absolute": float(np.mean(np.abs(
                    np.asarray(saved["ideal"]["positive_probabilities"])
                    - np.asarray(ideal["positive_probabilities"])
                ))),
                "max_absolute": float(np.max(np.abs(
                    np.asarray(saved["ideal"]["positive_probabilities"])
                    - np.asarray(ideal["positive_probabilities"])
                ))),
            },
        })

    trace = model["training"]["convergence"]
    if len(trace) != model["training"]["optimizer_evaluations"]:
        raise ValueError("saved optimizer trace length mismatch")
    result = {
        "schema_version": 1,
        "status": "verified from saved predictions and deterministic compilation; no model training",
        "dataset_identity_sha256": identity["dataset_sha256"],
        "model_file_sha256": model_sha,
        "original_artifact_sha256": {
            str(path.relative_to(ROOT)): _sha(path) for path in (
                BASELINE_PATH, IDEAL_PATH, BENCHMARK_PATH, MANIFEST_PATH, MODEL_PATH,
                TABLES / "historical" / "vqc_trained_parameters.json",
                TABLES / "historical" / "logistic_regression_baseline.json",
            )
        },
        "class_distribution": {
            "train_labels_0_1": np.bincount(dataset.y_train, minlength=2).tolist(),
            "test_labels_0_1": np.bincount(dataset.y_test, minlength=2).tolist(),
        },
        "preprocessing_check": "MinMaxScaler state matches only the 160 outer-training rows",
        "training_convergence": {
            "evaluation_count": len(trace),
            "initial_objective": float(trace[0]["objective"]),
            "final_objective": float(trace[-1]["objective"]),
            "best_recorded_objective": float(min(row["objective"] for row in trace)),
        },
        "exact_qnn_score_replay_max_absolute_error": float(np.max(np.abs(
            exact_scores - np.asarray(ideal["positive_probabilities"])
        ))),
        "final_layer_rz_perturbation_radians": 0.37,
        "final_layer_rz_max_score_changes": final_rz_sensitivity,
        "cases": cases,
        "logical_circuit": {
            "num_qubits": logical.num_qubits,
            "gate_only_depth": int(logical.depth(
                filter_function=lambda instruction: instruction.operation.name
                not in {"measure", "reset", "barrier"}
            )),
            "cx_count": int(logical.count_ops().get("cx", 0)),
        },
        "unconstrained_reference_cost": reference,
        "processors": topology_records,
        "noise_model_check": {
            "configured_rates": asdict(PRESCRIBED_NOISE),
            "noisy_operations": sorted(noise.noise_instructions),
            "symmetric_readout_matrix": readout["probabilities"],
        },
        "environment": {"qiskit": version("qiskit"), "qiskit_aer": version("qiskit-aer")},
    }
    _write(AUDIT_PATH, result)
    return result


def layout_ablation() -> dict[str, Any]:
    """Change only the initial layout strategy, keeping SABRE routing."""

    logical, _, _ = build_vqc_circuit(CONFIG.entanglement)
    reference = circuit_cost(compile_unconstrained(logical))
    saved = _load(BENCHMARK_PATH)
    rows: list[dict[str, Any]] = []
    for path, official in zip(PROCESSOR_PATHS, saved["processors"], strict=True):
        spec = ProcessorSpec.from_json(path)
        for layout_method in ("sabre", "trivial"):
            settings = {**COMPILER_SETTINGS, "layout_method": layout_method}
            expected_pass = "SabreLayout" if layout_method == "sabre" else "SabreSwap"
            events: list[dict[str, Any]] = []

            def callback(**kwargs: Any) -> None:
                pass_name = kwargs["pass_"].__class__.__name__
                if pass_name == expected_pass:
                    events.append({
                        "pass_name": pass_name,
                        "pass_index": int(kwargs["count"]),
                        "explicit_swap_count": int(kwargs["dag"].count_ops().get("swap", 0)),
                    })

            compiled = transpile(
                logical, coupling_map=CouplingMap(list(spec.coupling_edges)),
                callback=callback, **settings,
            )
            if len(events) != 1:
                raise RuntimeError(f"{spec.name} {layout_method}: expected one {expected_pass} event")
            cost = circuit_cost(compiled)
            if layout_method == "sabre" and cost != official["circuit_cost"]:
                raise ValueError(f"{spec.name} baseline cost changed")
            rows.append({
                "processor": spec.name,
                "layout_method": layout_method,
                "routing_method": "sabre",
                "compiler_settings": settings,
                "physical_qubit_budget": spec.num_qubits,
                "logical_qubit_count": logical.num_qubits,
                "logical_qubit_utilization_of_budget": logical.num_qubits / spec.num_qubits,
                "layout": layout_record(logical, compiled),
                "cost": cost,
                "routing_overhead": {
                    "gate_only_depth_delta": cost["gate_only_depth"] - reference["gate_only_depth"],
                    "cx_delta": cost["cx_count"] - reference["cx_count"],
                },
                "swap_trace_event": events[0],
            })
    result = {
        "schema_version": 1,
        "scope": "deterministic transpilation only; no training or simulation",
        "controlled_change": "initial layout_method sabre versus trivial; routing_method remains sabre",
        "logical_circuit": {"num_qubits": logical.num_qubits, "cx_count": 8},
        "unconstrained_reference_cost": reference,
        "qiskit_version": version("qiskit"),
        "model_architecture": "same fixed four-qubit ring/parity logical circuit for every row",
        "rows": rows,
    }
    _write(ABLATION_PATH, result)
    return result


if __name__ == "__main__":
    audit = audit_saved_results()
    ablation = layout_ablation()
    print(AUDIT_PATH, "verified", len(audit["cases"]), "classification cases")
    print(ABLATION_PATH, "recorded", len(ablation["rows"]), "transpilation cases")
