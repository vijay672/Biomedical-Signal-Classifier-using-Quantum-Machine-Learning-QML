"""One bounded provisional Q1 training run and official-settings A/B benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_aer import AerSimulator
from sklearn.linear_model import LogisticRegression

from src.artifacts import build_model_artifact, load_verified_model, save_model_artifact
from src.datasets import PreparedDataset, make_provisional_q1_benchmark
from src.metrics import classification_metrics
from src.problem import dataset_identity
from src.processor_benchmark import _bind_inputs, _predictions_from_result, make_noise_model
from src.processors import BASIS_GATES, NoiseRates, ProcessorSpec
from src.routing import COMPILER_SETTINGS, circuit_cost, compile_processor, compile_unconstrained, layout_record
from src.visualization import circuit_figure, coupling_figure, mapping_figure, results_figures
from src.vqc import VQCConfig, build_vqc_circuit, train_vqc_components


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODEL_PATH = TABLES / "q1_provisional_vqc_model.json"
BASELINE_PATH = TABLES / "q1_provisional_logistic.json"
IDEAL_PATH = TABLES / "q1_provisional_ideal_vqc.json"
BENCHMARK_PATH = TABLES / "q1_provisional_processor_benchmark.json"
MANIFEST_PATH = TABLES / "q1_provisional_manifest.json"
PROCESSOR_PATHS = (ROOT / "processors" / "processor_A.json", ROOT / "processors" / "processor_B.json")
PRESCRIBED_NOISE = NoiseRates(
    one_qubit_depolarizing=0.001,
    two_qubit_depolarizing=0.005,
    readout_flip=0.01,
)
SHOTS = 4096
CONFIG = VQCConfig(random_state=42, maxiter=25, shots=None, entanglement="ring", readout="parity")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _baseline(dataset: PreparedDataset) -> dict[str, Any]:
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(dataset.X_train, dataset.y_train)
    return {
        "schema_version": 2,
        "model": {"name": "LogisticRegression", "random_state": 42, "max_iter": 1000},
        "dataset_identity": dataset_identity(dataset),
        "metrics": classification_metrics(dataset.y_test, model.predict(dataset.X_test)),
        "environment": {"scikit_learn": version("scikit-learn"), "numpy": version("numpy")},
    }


def _processor_inference(
    dataset: PreparedDataset,
    model_path: Path,
) -> tuple[dict[str, Any], list[tuple[ProcessorSpec, Any]]]:
    weights, model_sha256 = load_verified_model(model_path, dataset, CONFIG)
    logical, inputs, parameters = build_vqc_circuit(CONFIG.entanglement)
    unconstrained = compile_unconstrained(logical)
    reference_cost = circuit_cost(unconstrained)
    ideal_sim = AerSimulator(method="density_matrix", max_parallel_threads=1)
    noisy_sim = AerSimulator(
        method="density_matrix", noise_model=make_noise_model(PRESCRIBED_NOISE),
        max_parallel_threads=1,
    )
    records: list[dict[str, Any]] = []
    circuits: list[tuple[ProcessorSpec, Any]] = []
    for path in PROCESSOR_PATHS:
        spec = ProcessorSpec.from_json(path)
        compiled = compile_processor(logical, spec)
        circuits.append((spec, compiled))
        cost = circuit_cost(compiled)
        bound = _bind_inputs(compiled, inputs, parameters, weights, dataset.X_test)
        modes: dict[str, Any] = {}
        for mode, simulator in (("ideal", ideal_sim), ("noisy", noisy_sim)):
            result = simulator.run(bound, shots=SHOTS, seed_simulator=42).result()
            if not result.success:
                raise RuntimeError(f"{spec.name} {mode} Aer simulation failed")
            predictions, probabilities = _predictions_from_result(result, len(dataset.y_test))
            modes[mode] = {
                "metrics": classification_metrics(dataset.y_test, np.asarray(predictions)),
                "positive_probabilities": probabilities,
            }
        records.append({
            "name": spec.name,
            "processor_file": str(path.relative_to(ROOT)),
            "num_qubits": spec.num_qubits,
            "coupling_map_source": spec.coupling_map_source,
            "design_rationale": json.loads(path.read_text())["design_rationale"],
            "coupling_edges": [list(edge) for edge in spec.coupling_edges],
            "layout": layout_record(logical, compiled),
            "circuit_cost": cost,
            "routing_overhead": {
                "gate_only_depth_delta": cost["gate_only_depth"] - reference_cost["gate_only_depth"],
                "full_measured_depth_delta": cost["full_measured_depth"] - reference_cost["full_measured_depth"],
                "cx_delta": cost["cx_count"] - reference_cost["cx_count"],
                "definition": "difference from unconstrained compile with same basis, level, seed",
            },
            **modes,
        })
    return {
        "schema_version": 2,
        "benchmark": "provisional Q1 synthetic fixture; prescribed Phase 1 noise",
        "dataset_identity": dataset_identity(dataset),
        "model_file": str(model_path.relative_to(ROOT)),
        "model_file_sha256": model_sha256,
        "shared_settings": {
            "basis_gates": list(BASIS_GATES),
            "optimization_level": 1,
            "layout_method": "sabre",
            "routing_method": "sabre",
            "seed_transpiler": 42,
            "seed_simulator": 42,
            "shots_per_circuit": SHOTS,
            "noise_rates": asdict(PRESCRIBED_NOISE),
            "noise_model": "uniform depolarizing on rz/sx/x and cx; symmetric readout bit-flip",
        },
        "unconstrained_circuit_cost": reference_cost,
        "swap_count_note": "native-basis transpilation decomposes routing; exact inserted SWAP count is not verified",
        "environment": {"qiskit": version("qiskit"), "qiskit_aer": version("qiskit-aer"),
                        "numpy": version("numpy"), "scipy": version("scipy")},
        "processors": records,
    }, circuits


def smoke_test() -> dict[str, Any]:
    """Check data, QNN output and explicit compilation without training."""

    dataset = make_provisional_q1_benchmark()
    logical, _, _ = build_vqc_circuit()
    specs = [ProcessorSpec.from_json(path) for path in PROCESSOR_PATHS]
    compiled = [compile_processor(logical, spec) for spec in specs]
    if dataset.X_train.shape != (160, 4) or dataset.X_test.shape != (40, 4):
        raise RuntimeError("unexpected provisional dataset shape")
    return {"dataset_sha256": dataset_identity(dataset)["dataset_sha256"],
            "cx_counts": [circuit_cost(c)["cx_count"] for c in compiled],
            "noise_rates": asdict(PRESCRIBED_NOISE)}


def run_all() -> dict[str, Any]:
    """Train the corrected fixture once, then benchmark its saved model on A/B."""

    dataset = make_provisional_q1_benchmark()
    identity = dataset_identity(dataset)
    baseline = _baseline(dataset)
    _write(BASELINE_PATH, baseline)

    convergence: list[dict[str, float | int]] = []
    def callback(_weights: np.ndarray, objective: float) -> None:
        convergence.append({"evaluation": len(convergence) + 1, "objective": float(objective)})

    components = train_vqc_components(dataset.X_train, dataset.y_train, CONFIG, callback=callback)
    fit = components.classifier.fit_result
    if not convergence:
        raise RuntimeError("COBYLA did not produce a convergence trace")
    artifact = build_model_artifact(
        dataset, np.asarray(components.classifier.weights, dtype=float), convergence,
        config=CONFIG, optimizer_evaluations=int(fit.nfev), final_objective=float(fit.fun),
    )
    save_model_artifact(artifact, MODEL_PATH)
    # The same saved and validated weights are used for ideal and both processor runs.
    verified_weights, _ = load_verified_model(MODEL_PATH, dataset, CONFIG)
    logical, inputs, parameters = build_vqc_circuit()
    from qiskit_machine_learning.primitives import QMLSampler
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from src.vqc import parity_class
    qnn = SamplerQNN(circuit=logical, sampler=QMLSampler(shots=None, seed=42),
                     input_params=inputs, weight_params=parameters,
                     interpret=parity_class, output_shape=2)
    probabilities = np.asarray(qnn.forward(dataset.X_test, verified_weights), dtype=float)
    if probabilities.shape != (40, 2):
        raise RuntimeError("ideal QNN output has unexpected shape")
    ideal = {
        "schema_version": 2,
        "dataset_sha256": identity["dataset_sha256"],
        "model_file": str(MODEL_PATH.relative_to(ROOT)),
        "metrics": classification_metrics(dataset.y_test, np.argmax(probabilities, axis=1)),
        "positive_probabilities": probabilities[:, 1].tolist(),
        "simulator": "QMLSampler exact statevector, shots=None, seed=42",
    }
    _write(IDEAL_PATH, ideal)
    benchmark, circuits = _processor_inference(dataset, MODEL_PATH)
    _write(BENCHMARK_PATH, benchmark)

    figure_paths: list[Path] = []
    figure_paths.extend(results_figures(baseline, ideal, benchmark, convergence, FIGURES))
    logical_path = FIGURES / "logical_circuit.svg"
    circuit_figure(logical, "Four-qubit RY / two-layer ring VQC (logical)", logical_path)
    figure_paths.append(logical_path)
    (TABLES / "q1_provisional_logical_circuit.txt").write_text(
        str(logical.draw(output="text", fold=100)) + "\n", encoding="utf-8"
    )
    for spec, compiled in circuits:
        suffix = "A" if spec.num_qubits == 5 else "B"
        record = next(item for item in benchmark["processors"] if item["name"] == spec.name)
        coupling = FIGURES / f"processor_{suffix}_coupling.svg"
        transpiled = FIGURES / f"processor_{suffix}_transpiled_circuit.svg"
        coupling_figure(spec, record["layout"], coupling)
        circuit_figure(compiled, f"{spec.name} transpiled (SABRE, basis rz/sx/x/cx)", transpiled)
        figure_paths.extend((coupling, transpiled))
        (TABLES / f"q1_provisional_processor_{suffix}_transpiled_circuit.txt").write_text(
            str(compiled.draw(output="text", fold=100)) + "\n", encoding="utf-8"
        )
    mapping = FIGURES / "logical_to_physical_mapping.svg"
    mapping_figure(benchmark["processors"], mapping)
    figure_paths.append(mapping)

    manifest = {
        "schema_version": 2,
        "problem_code": "PS-Q1",
        "problem_title_master_guide": "Variational Quantum Classifier (VQC)",
        "dataset_status": "provisional; n_redundant=0 is an engineering assumption",
        "dataset_identity": identity,
        "training_runs": 1,
        "selection_rule": "fixed original ring/parity architecture; no held-out model selection",
        "baseline_file": str(BASELINE_PATH.relative_to(ROOT)),
        "model_file": str(MODEL_PATH.relative_to(ROOT)),
        "ideal_file": str(IDEAL_PATH.relative_to(ROOT)),
        "processor_file": str(BENCHMARK_PATH.relative_to(ROOT)),
        "figure_files": [str(path.relative_to(ROOT)) for path in figure_paths],
        "environment": artifact["environment"],
    }
    _write(MANIFEST_PATH, manifest)
    return {"baseline": baseline, "ideal": ideal, "benchmark": benchmark,
            "manifest": manifest, "convergence": convergence}


def load_existing_or_run() -> dict[str, Any]:
    """Reuse a complete, identity-checked run; train only when no run exists."""

    if not MANIFEST_PATH.exists():
        return run_all()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dataset = make_provisional_q1_benchmark()
    identity = dataset_identity(dataset)
    if manifest.get("dataset_identity") != identity:
        raise ValueError("saved manifest does not match the provisional dataset")
    _, model_hash = load_verified_model(MODEL_PATH, dataset, CONFIG)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    ideal = json.loads(IDEAL_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    if baseline.get("dataset_identity") != identity:
        raise ValueError("saved classical baseline uses a different dataset or split")
    if ideal.get("dataset_sha256") != identity["dataset_sha256"]:
        raise ValueError("saved ideal VQC uses a different dataset or split")
    if (benchmark.get("dataset_identity") != identity or
            benchmark.get("model_file_sha256") != model_hash or
            len(benchmark.get("processors", [])) != 2):
        raise ValueError("saved processor benchmark uses different data or weights")
    if benchmark.get("shared_settings", {}).get("noise_rates") != asdict(PRESCRIBED_NOISE):
        raise ValueError("saved processor benchmark does not use prescribed noise")
    for figure in manifest.get("figure_files", []):
        if not (ROOT / figure).is_file():
            raise FileNotFoundError(f"saved figure missing: {figure}")
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return {"baseline": baseline, "ideal": ideal, "benchmark": benchmark,
            "manifest": manifest, "convergence": artifact["training"]["convergence"]}


if __name__ == "__main__":
    output = load_existing_or_run()
    print("Verified provisional Q1 benchmark:", MANIFEST_PATH)
    for name in ("baseline", "ideal"):
        print(name, output[name]["metrics"])
    for processor in output["benchmark"]["processors"]:
        print(processor["name"], "noisy", processor["noisy"]["metrics"])
