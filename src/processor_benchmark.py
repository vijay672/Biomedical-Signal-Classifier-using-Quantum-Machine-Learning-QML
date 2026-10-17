"""Historical original-fixture A/B run with alternative noise.

The official corrected Phase 1 entry point is ``python -m src.q1_submission``.
This module preserves the earlier 0.001/0.01/0.02 experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from sklearn.metrics import confusion_matrix, recall_score

from src.datasets import PreparedDataset, make_synthetic_benchmark
from src.vqc import DEFAULT_PARAMETERS_PATH, NUM_QUBITS, build_vqc_circuit, parity_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_PATH = PROJECT_ROOT / "results" / "processor_benchmark.json"
PROCESSOR_FILES = (
    PROJECT_ROOT / "processors" / "processor_A.json",
    PROJECT_ROOT / "processors" / "processor_B.json",
)
BASIS_GATES = ("rz", "sx", "x", "cx")


@dataclass(frozen=True)
class ProcessorSpec:
    name: str
    num_qubits: int
    coupling_map_source: str
    reference_url: str | None
    topology: str
    coupling_edges: tuple[tuple[int, int], ...]

    @classmethod
    def from_json(cls, path: Path | str) -> ProcessorSpec:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        edges = tuple(tuple(edge) for edge in payload["coupling_edges"])
        spec = cls(
            name=payload["name"],
            num_qubits=int(payload["num_qubits"]),
            coupling_map_source=payload["coupling_map_source"],
            reference_url=payload.get("reference_url"),
            topology=payload["topology"],
            coupling_edges=edges,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.num_qubits < NUM_QUBITS:
            raise ValueError("processor must have at least four physical qubits")
        if self.coupling_map_source not in {"custom", "reference"}:
            raise ValueError("coupling_map_source must be 'custom' or 'reference'")
        if self.coupling_map_source == "reference" and not self.reference_url:
            raise ValueError("reference coupling maps require a source URL")
        if self.coupling_map_source == "custom" and self.reference_url is not None:
            raise ValueError("custom coupling maps must not claim a reference URL")
        if not self.coupling_edges or len(set(self.coupling_edges)) != len(self.coupling_edges):
            raise ValueError("coupling map edges must be non-empty and unique")
        for edge in self.coupling_edges:
            if (
                len(edge) != 2
                or not all(isinstance(qubit, int) for qubit in edge)
                or edge[0] == edge[1]
                or not all(0 <= qubit < self.num_qubits for qubit in edge)
            ):
                raise ValueError(f"invalid coupling map edge: {edge}")
        used_qubits = {qubit for edge in self.coupling_edges for qubit in edge}
        if used_qubits != set(range(self.num_qubits)):
            raise ValueError("coupling map must include every physical qubit")
        if not CouplingMap(list(self.coupling_edges)).is_connected():
            raise ValueError("coupling map must be connected")


@dataclass(frozen=True)
class NoiseRates:
    one_qubit_depolarizing: float = 0.001
    two_qubit_depolarizing: float = 0.01
    readout_flip: float = 0.02

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class BenchmarkSettings:
    basis_gates: tuple[str, ...] = BASIS_GATES
    optimization_level: int = 1
    seed: int = 42
    shots: int = 4096
    noise: NoiseRates = NoiseRates()

    def __post_init__(self) -> None:
        if not 0 <= self.optimization_level <= 3:
            raise ValueError("optimization_level must be between 0 and 3")
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if set(self.basis_gates) != set(BASIS_GATES):
            raise ValueError(f"basis_gates must contain exactly {BASIS_GATES}")


def load_trained_weights(
    path: Path | str,
    dataset: PreparedDataset,
    expected_parameters: tuple[Parameter, ...],
) -> tuple[np.ndarray, str]:
    """Check the saved VQC and dataset identity before processor simulation."""

    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if payload.get("architecture", {}).get("num_qubits") != NUM_QUBITS:
        raise ValueError("saved VQC does not have the expected four-qubit architecture")
    metadata = payload.get("dataset", {})
    expected_data = {
        "source": dataset.source,
        "train_samples": len(dataset.y_train),
        "test_samples": len(dataset.y_test),
        "features": dataset.X_test.shape[1],
        "generation_random_state": dataset.generation_random_state,
        "split_random_state": dataset.random_state,
        "test_size": dataset.test_size,
    }
    for name, expected in expected_data.items():
        if metadata.get(name) != expected:
            raise ValueError(f"saved VQC dataset mismatch on {name}")

    saved_parameters = payload.get("trained_parameters", [])
    names = [entry.get("name") for entry in saved_parameters]
    if names != [str(parameter) for parameter in expected_parameters]:
        raise ValueError("saved VQC parameter names or ordering do not match the circuit")
    weights = np.asarray([entry["value_radians"] for entry in saved_parameters], dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("saved VQC has non-finite parameter values")
    return weights, hashlib.sha256(raw).hexdigest()


def make_noise_model(rates: NoiseRates) -> NoiseModel:
    """Use one uniform synthetic noise model for both processor topologies."""

    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(rates.one_qubit_depolarizing, 1),
        ["rz", "sx", "x"],
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(rates.two_qubit_depolarizing, 2),
        ["cx"],
    )
    readout = rates.readout_flip
    model.add_all_qubit_readout_error(
        ReadoutError([[1 - readout, readout], [readout, 1 - readout]])
    )
    return model


def _circuit_stats(circuit: QuantumCircuit) -> dict[str, int]:
    return {"cx_count": int(circuit.count_ops().get("cx", 0)), "depth": int(circuit.depth())}


def _bind_inputs(
    compiled: QuantumCircuit,
    input_parameters: tuple[Parameter, ...],
    weight_parameters: tuple[Parameter, ...],
    weights: np.ndarray,
    X: np.ndarray,
) -> list[QuantumCircuit]:
    weighted = compiled.assign_parameters(
        dict(zip(weight_parameters, weights, strict=True)), inplace=False
    )
    return [
        weighted.assign_parameters(dict(zip(input_parameters, row, strict=True)), inplace=False)
        for row in X
    ]


def _predictions_from_result(result: Any, count: int) -> tuple[list[int], list[float]]:
    predictions: list[int] = []
    positive_probabilities: list[float] = []
    for index in range(count):
        counts = result.get_counts(index)
        total = sum(counts.values())
        positive = sum(
            occurrences
            for bits, occurrences in counts.items()
            if parity_class(int(bits.replace(" ", ""), 2)) == 1
        )
        probability = positive / total
        positive_probabilities.append(probability)
        predictions.append(int(probability > 0.5))
    return predictions, positive_probabilities


def _prediction_summary(
    predictions: list[int], probabilities: list[float], y_true: np.ndarray
) -> dict[str, Any]:
    return {
        "predictions": predictions,
        "positive_probabilities": probabilities,
        "recall_label_1": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def _evaluate_processor(
    spec: ProcessorSpec,
    circuit: QuantumCircuit,
    input_parameters: tuple[Parameter, ...],
    weight_parameters: tuple[Parameter, ...],
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    reference_stats: dict[str, int],
    settings: BenchmarkSettings,
    ideal_simulator: AerSimulator,
    noisy_simulator: AerSimulator,
) -> dict[str, Any]:
    compiled = transpile(
        circuit,
        coupling_map=CouplingMap(list(spec.coupling_edges)),
        basis_gates=list(settings.basis_gates),
        optimization_level=settings.optimization_level,
        seed_transpiler=settings.seed,
    )
    stats = _circuit_stats(compiled)
    bound = _bind_inputs(compiled, input_parameters, weight_parameters, weights, X)
    ideal_result = ideal_simulator.run(
        bound, shots=settings.shots, seed_simulator=settings.seed
    ).result()
    noisy_result = noisy_simulator.run(
        bound, shots=settings.shots, seed_simulator=settings.seed
    ).result()
    if not ideal_result.success or not noisy_result.success:
        raise RuntimeError(f"Aer simulation failed for {spec.name}")
    ideal_pred, ideal_prob = _predictions_from_result(ideal_result, len(X))
    noisy_pred, noisy_prob = _predictions_from_result(noisy_result, len(X))
    ideal = _prediction_summary(ideal_pred, ideal_prob, y)
    noisy = _prediction_summary(noisy_pred, noisy_prob, y)
    return {
        "name": spec.name,
        "num_qubits": spec.num_qubits,
        "coupling_map_source": spec.coupling_map_source,
        "reference_url": spec.reference_url,
        "topology": spec.topology,
        "coupling_edges": [list(edge) for edge in spec.coupling_edges],
        "transpiled_qubits": compiled.num_qubits,
        "cx_count": stats["cx_count"],
        "depth": stats["depth"],
        "routing_overhead_proxy": {
            "cx_delta_vs_unconstrained": stats["cx_count"] - reference_stats["cx_count"],
            "depth_delta_vs_unconstrained": stats["depth"] - reference_stats["depth"],
        },
        "ideal": ideal,
        "noisy": noisy,
        "noisy_minus_ideal_recall": noisy["recall_label_1"] - ideal["recall_label_1"],
        "changed_predictions": sum(a != b for a, b in zip(ideal_pred, noisy_pred, strict=True)),
    }


def run_benchmark(
    *,
    parameters_path: Path | str = DEFAULT_PARAMETERS_PATH,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    processor_files: tuple[Path | str, Path | str] = PROCESSOR_FILES,
    settings: BenchmarkSettings = BenchmarkSettings(),
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate the same saved VQC on both custom Aer processor models."""

    dataset = make_synthetic_benchmark()
    circuit, inputs, parameter_names = build_vqc_circuit()
    weights, parameter_sha256 = load_trained_weights(
        parameters_path, dataset, parameter_names
    )
    specs = tuple(ProcessorSpec.from_json(path) for path in processor_files)
    if sorted(spec.num_qubits for spec in specs) != [5, 9]:
        raise ValueError("benchmark requires one five-qubit and one nine-qubit processor")
    if sample_limit is not None and not 1 <= sample_limit <= len(dataset.y_test):
        raise ValueError("sample_limit must select 1 to 40 held-out rows")
    X = dataset.X_test if sample_limit is None else dataset.X_test[:sample_limit]
    y = dataset.y_test if sample_limit is None else dataset.y_test[:sample_limit]

    unconstrained = transpile(
        circuit,
        basis_gates=list(settings.basis_gates),
        optimization_level=settings.optimization_level,
        seed_transpiler=settings.seed,
    )
    reference_stats = _circuit_stats(unconstrained)
    noise_model = make_noise_model(settings.noise)
    ideal_simulator = AerSimulator(method="density_matrix", max_parallel_threads=1)
    noisy_simulator = AerSimulator(
        method="density_matrix", noise_model=noise_model, max_parallel_threads=1
    )
    processors = [
        _evaluate_processor(
            spec, circuit, inputs, parameter_names, weights, X, y,
            reference_stats, settings, ideal_simulator, noisy_simulator,
        )
        for spec in specs
    ]
    dataset_hash = hashlib.sha256(
        np.ascontiguousarray(dataset.X_test, dtype="<f8").tobytes()
        + np.ascontiguousarray(dataset.y_test, dtype="<i8").tobytes()
    ).hexdigest()
    resolved_parameters_path = Path(parameters_path).resolve()
    try:
        parameter_file_label = str(resolved_parameters_path.relative_to(PROJECT_ROOT))
    except ValueError:
        parameter_file_label = str(resolved_parameters_path)
    payload = {
        "schema_version": 1,
        "map_choice": (
            "custom illustrative maps; no architecture/processor reference map was available "
            "in the repository, and organizer coupling-map authority remains unresolved"
        ),
        "settings_shared_by_processors": {
            "basis_gates": list(settings.basis_gates),
            "optimization_level": settings.optimization_level,
            "seed_transpiler_and_simulator": settings.seed,
            "shots_per_circuit": settings.shots,
            "noise_rates": asdict(settings.noise),
            "noise_model": "uniform depolarizing on rz/sx/x and cx, symmetric bit-flip readout",
        },
        "dataset": {
            "source": dataset.source,
            "generation_random_state": dataset.generation_random_state,
            "split_random_state": dataset.random_state,
            "test_size": dataset.test_size,
            "full_test_samples": len(dataset.y_test),
            "evaluated_samples": len(y),
            "test_partition_sha256": dataset_hash,
            "true_labels": y.tolist(),
        },
        "trained_vqc": {
            "parameter_file": parameter_file_label,
            "parameter_file_sha256": parameter_sha256,
            "parameter_count": len(weights),
        },
        "unconstrained_compile_baseline": reference_stats,
        "routing_overhead_definition": (
            "transpiled CX count and depth minus an unconstrained four-qubit compile "
            "of the same circuit, basis, optimization level, and seed"
        ),
        "environment": {
            "qiskit": version("qiskit"),
            "qiskit_aer": version("qiskit-aer"),
            "scikit_learn": version("scikit-learn"),
        },
        "processors": processors,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print("Historical original-fixture/alternative-noise run; official entry point: src.q1_submission")
    report = run_benchmark()
    for processor in report["processors"]:
        print(
            f"{processor['name']}: CX={processor['cx_count']}, depth={processor['depth']}, "
            f"ideal recall={processor['ideal']['recall_label_1']:.3f}, "
            f"noisy recall={processor['noisy']['recall_label_1']:.3f}"
        )
    print(f"Saved {DEFAULT_RESULT_PATH}")
