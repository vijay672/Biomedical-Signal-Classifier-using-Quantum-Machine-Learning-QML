"""Strict, versioned artifact format for the provisional Q1 benchmark."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets import PreparedDataset
from src.problem import dataset_identity
from src.vqc import NUM_LAYERS, NUM_QUBITS, VQCConfig, build_vqc_circuit


MODEL_SCHEMA_VERSION = 2
ARCHITECTURE_VERSION = "q1-ry-ring-parity-2layer-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def architecture_identity(config: VQCConfig) -> dict[str, Any]:
    if config.entanglement != "ring" or config.readout != "parity":
        raise ValueError("official Q1 artifact requires the ring/parity architecture")
    circuit, inputs, weights = build_vqc_circuit(config.entanglement)
    definition = {
        "architecture_version": ARCHITECTURE_VERSION,
        "num_qubits": NUM_QUBITS,
        "input_encoding": "RY angle encoding, feature i on logical qubit i",
        "input_parameters": [str(p) for p in inputs],
        "trainable_layers": NUM_LAYERS,
        "rotations_per_qubit_per_layer": ["RY", "RZ"],
        "entanglement": "directed ring CX",
        "entanglement_edges_per_layer": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "readout": "parity of four measured logical bits; label 1 is odd parity",
        "parameter_names": [str(p) for p in weights],
        "parameter_count": len(weights),
        "logical_operation_counts": {str(k): int(v) for k, v in circuit.count_ops().items()},
    }
    return {"definition": definition, "architecture_sha256": _digest(definition)}


def build_model_artifact(
    dataset: PreparedDataset,
    weights: np.ndarray,
    convergence: list[dict[str, float | int]],
    *,
    config: VQCConfig = VQCConfig(),
    optimizer_evaluations: int,
    final_objective: float,
) -> dict[str, Any]:
    architecture = architecture_identity(config)
    names = architecture["definition"]["parameter_names"]
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("trained parameter count or values are invalid")
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "architecture": architecture,
        "dataset_identity": dataset_identity(dataset),
        "trained_parameters": [
            {"name": name, "value_radians": float(value)}
            for name, value in zip(names, values, strict=True)
        ],
        "training": {
            "optimizer": "COBYLA",
            "optimizer_evaluations": optimizer_evaluations,
            "maxiter": config.maxiter,
            "random_state": config.random_state,
            "simulator": "QMLSampler exact statevector, shots=None",
            "final_objective": final_objective,
            "convergence": convergence,
        },
        "environment": {
            "qiskit": version("qiskit"),
            "qiskit_aer": version("qiskit-aer"),
            "qiskit_machine_learning": version("qiskit-machine-learning"),
            "scikit_learn": version("scikit-learn"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
    }


def save_model_artifact(payload: dict[str, Any], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_verified_model(
    path: Path | str,
    dataset: PreparedDataset,
    config: VQCConfig = VQCConfig(),
) -> tuple[np.ndarray, str]:
    """Reject architecture, split, scaler, version, or parameter mismatch."""

    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("model artifact schema version mismatch")
    if payload.get("architecture") != architecture_identity(config):
        raise ValueError("model architecture signature mismatch")
    if payload.get("dataset_identity") != dataset_identity(dataset):
        raise ValueError("model dataset, split, or preprocessing identity mismatch")
    names = payload["architecture"]["definition"]["parameter_names"]
    saved = payload.get("trained_parameters", [])
    if [entry.get("name") for entry in saved] != names:
        raise ValueError("model parameter names or ordering mismatch")
    weights = np.asarray([entry["value_radians"] for entry in saved], dtype=float)
    if weights.shape != (len(names),) or not np.isfinite(weights).all():
        raise ValueError("model parameter dimension or values mismatch")
    expected_versions = {
        "qiskit": version("qiskit"),
        "qiskit_aer": version("qiskit-aer"),
        "qiskit_machine_learning": version("qiskit-machine-learning"),
        "scikit_learn": version("scikit-learn"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }
    if payload.get("environment") != expected_versions:
        raise ValueError("model software-version mismatch")
    return weights, hashlib.sha256(raw).hexdigest()
