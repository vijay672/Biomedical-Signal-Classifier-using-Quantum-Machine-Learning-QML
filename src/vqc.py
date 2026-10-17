"""Four-qubit variational classifier on the shared Q1 dataset pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.primitives import QMLSampler

from src.datasets import PreparedDataset, make_synthetic_benchmark


NUM_QUBITS = 4
NUM_LAYERS = 2
WEIGHTS_PER_LAYER = NUM_QUBITS * 2  # RY and RZ angle per qubit.
NUM_WEIGHTS = NUM_LAYERS * WEIGHTS_PER_LAYER
MIN_COBYLA_EVALUATIONS = NUM_WEIGHTS + 2
DEFAULT_MAXITER = 25
MAX_ALLOWED_ITERATIONS = 100
DEFAULT_PARAMETERS_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "vqc_trained_parameters.json"
)


@dataclass(frozen=True)
class VQCConfig:
    """Reproducible simulator and bounded optimizer settings."""

    random_state: int = 42
    maxiter: int = DEFAULT_MAXITER
    shots: int | None = None
    entanglement: str = "ring"
    readout: str = "parity"

    def __post_init__(self) -> None:
        if not MIN_COBYLA_EVALUATIONS <= self.maxiter <= MAX_ALLOWED_ITERATIONS:
            raise ValueError(
                "maxiter must be between "
                f"{MIN_COBYLA_EVALUATIONS} and {MAX_ALLOWED_ITERATIONS} "
                "for this parameter count"
            )
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive or None for exact simulation")
        if self.entanglement not in {"ring", "line"}:
            raise ValueError("entanglement must be 'ring' or 'line'")
        if self.readout not in {"parity", "first_qubit"}:
            raise ValueError("readout must be 'parity' or 'first_qubit'")


@dataclass(frozen=True)
class VQCComponents:
    circuit: QuantumCircuit
    input_parameters: tuple[Any, ...]
    weight_parameters: tuple[Any, ...]
    qnn: SamplerQNN
    classifier: NeuralNetworkClassifier
    initial_weights: np.ndarray


@dataclass(frozen=True)
class VQCTrainingResult:
    components: VQCComponents
    config: VQCConfig
    trained_weights: np.ndarray
    test_probabilities: np.ndarray
    test_predictions: np.ndarray
    test_accuracy: float
    optimizer_evaluations: int
    final_objective: float


def build_vqc_circuit(
    entanglement: str = "ring",
) -> tuple[QuantumCircuit, tuple[Any, ...], tuple[Any, ...]]:
    """Create RY encoding and two trainable RY/RZ plus CNOT layers."""

    if entanglement not in {"ring", "line"}:
        raise ValueError("entanglement must be 'ring' or 'line'")

    circuit = QuantumCircuit(NUM_QUBITS, name="qff_four_qubit_vqc")
    inputs = ParameterVector("x", NUM_QUBITS)
    weights = ParameterVector("w", NUM_WEIGHTS)

    for qubit in range(NUM_QUBITS):
        circuit.ry(inputs[qubit], qubit)

    for layer in range(NUM_LAYERS):
        layer_offset = layer * WEIGHTS_PER_LAYER
        for qubit in range(NUM_QUBITS):
            weight_offset = layer_offset + 2 * qubit
            circuit.ry(weights[weight_offset], qubit)
            circuit.rz(weights[weight_offset + 1], qubit)
        for qubit in range(NUM_QUBITS if entanglement == "ring" else NUM_QUBITS - 1):
            circuit.cx(qubit, (qubit + 1) % NUM_QUBITS)

    circuit.measure_all()
    return circuit, tuple(inputs), tuple(weights)


def parity_class(bitstring: int) -> int:
    """Map a sampled four-bit outcome to one of two parity classes."""

    return int(bitstring).bit_count() % 2


def first_qubit_class(bitstring: int) -> int:
    """Map a measured outcome to the state of logical qubit zero."""

    return int(bitstring) & 1


def build_vqc_components(
    config: VQCConfig = VQCConfig(),
    callback: Callable[[np.ndarray, float], None] | None = None,
) -> VQCComponents:
    """Build SamplerQNN and NeuralNetworkClassifier for an ideal simulator."""

    circuit, input_parameters, weight_parameters = build_vqc_circuit(config.entanglement)
    sampler = QMLSampler(shots=config.shots, seed=config.random_state)
    qnn = SamplerQNN(
        circuit=circuit,
        sampler=sampler,
        input_params=input_parameters,
        weight_params=weight_parameters,
        interpret=parity_class if config.readout == "parity" else first_qubit_class,
        output_shape=2,
    )
    initial_weights = np.random.default_rng(config.random_state).uniform(
        low=-np.pi,
        high=np.pi,
        size=NUM_WEIGHTS,
    )
    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        loss="cross_entropy",
        one_hot=True,
        optimizer=COBYLA(maxiter=config.maxiter, tol=1e-3, rhobeg=0.5),
        initial_point=initial_weights.copy(),
        callback=callback,
    )
    return VQCComponents(
        circuit=circuit,
        input_parameters=input_parameters,
        weight_parameters=weight_parameters,
        qnn=qnn,
        classifier=classifier,
        initial_weights=initial_weights,
    )


def train_vqc_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: VQCConfig = VQCConfig(),
    callback: Callable[[np.ndarray, float], None] | None = None,
) -> VQCComponents:
    """Fit one configured VQC using only the supplied training rows."""

    if X_train.ndim != 2 or X_train.shape[1] != NUM_QUBITS:
        raise ValueError(f"the four-qubit VQC requires {NUM_QUBITS} input features")
    if y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
        raise ValueError("one binary label is required per training row")
    components = build_vqc_components(config, callback=callback)
    components.classifier.fit(X_train, np.eye(2, dtype=np.float64)[y_train])
    return components


def fit_vqc(
    dataset: PreparedDataset,
    config: VQCConfig = VQCConfig(),
) -> VQCTrainingResult:
    """Train on the supplied training partition and predict its held-out set."""

    if dataset.X_train.ndim != 2 or dataset.X_train.shape[1] != NUM_QUBITS:
        raise ValueError(f"the four-qubit VQC requires {NUM_QUBITS} input features")
    components = train_vqc_components(dataset.X_train, dataset.y_train, config)

    probabilities = np.asarray(components.classifier.predict_proba(dataset.X_test))
    if probabilities.shape != (dataset.X_test.shape[0], 2):
        raise RuntimeError(
            "SamplerQNN classifier returned an unexpected probability shape: "
            f"{probabilities.shape}"
        )
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    fit_result = components.classifier.fit_result
    return VQCTrainingResult(
        components=components,
        config=config,
        trained_weights=np.asarray(components.classifier.weights, dtype=np.float64),
        test_probabilities=probabilities,
        test_predictions=predictions,
        test_accuracy=float(np.mean(predictions == dataset.y_test)),
        optimizer_evaluations=int(getattr(fit_result, "nfev", 0)),
        final_objective=float(fit_result.fun),
    )


def save_vqc_parameters(
    result: VQCTrainingResult,
    dataset: PreparedDataset,
    path: Path | str = DEFAULT_PARAMETERS_PATH,
) -> Path:
    """Save optimized rotation weights and sufficient run metadata as JSON."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "architecture": {
            "num_qubits": NUM_QUBITS,
            "input_encoding": "RY angle encoding",
            "trainable_layers": NUM_LAYERS,
            "trainable_rotations_per_qubit_per_layer": ["RY", "RZ"],
            "entanglement": (
                "directed nearest-neighbor ring of CNOT gates"
                if result.config.entanglement == "ring"
                else "directed nearest-neighbor line of CNOT gates"
            ),
            "readout": (
                "parity of measured four-bit outcomes"
                if result.config.readout == "parity"
                else "state of logical qubit zero"
            ),
        },
        "trained_parameters": [
            {"name": str(parameter), "value_radians": float(value)}
            for parameter, value in zip(
                result.components.weight_parameters, result.trained_weights, strict=True
            )
        ],
        "training": {
            "optimizer": "COBYLA",
            "maxiter": result.config.maxiter,
            "optimizer_evaluations": result.optimizer_evaluations,
            "final_objective": result.final_objective,
            "random_state": result.config.random_state,
            "initial_weights_radians": result.components.initial_weights.tolist(),
        },
        "simulator": {
            "name": "Qiskit Machine Learning QMLSampler",
            "mode": "exact statevector" if result.config.shots is None else "sampled statevector",
            "shots": result.config.shots,
            "seed": result.config.random_state,
        },
        "environment": {
            "qiskit": version("qiskit"),
            "qiskit_aer": version("qiskit-aer"),
            "qiskit_machine_learning": version("qiskit-machine-learning"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "dataset": {
            "source": dataset.source,
            "train_samples": int(dataset.X_train.shape[0]),
            "test_samples": int(dataset.X_test.shape[0]),
            "features": int(dataset.X_train.shape[1]),
            "generation_random_state": dataset.generation_random_state,
            "split_random_state": dataset.random_state,
            "test_size": dataset.test_size,
            "preprocessing": "shared MinMaxScaler fitted on training rows only to [0, pi]",
        },
        "held_out": {
            "accuracy": result.test_accuracy,
            "probability_shape": list(result.test_probabilities.shape),
            "prediction_shape": list(result.test_predictions.shape),
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def run_synthetic_vqc(
    output_path: Path | str = DEFAULT_PARAMETERS_PATH,
    config: VQCConfig = VQCConfig(),
) -> VQCTrainingResult:
    """Train the default synthetic Q1 VQC and save its trained parameters."""

    dataset = make_synthetic_benchmark()
    result = fit_vqc(dataset, config)
    save_vqc_parameters(result, dataset, output_path)
    return result


if __name__ == "__main__":
    result = run_synthetic_vqc()
    print(json.dumps({
        "test_accuracy": result.test_accuracy,
        "test_samples": int(result.test_predictions.shape[0]),
        "trained_parameter_count": int(result.trained_weights.shape[0]),
        "optimizer_evaluations": result.optimizer_evaluations,
        "saved_to": str(DEFAULT_PARAMETERS_PATH),
    }, indent=2))
