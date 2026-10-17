"""Validation for custom-map Aer processor comparisons."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from sklearn.metrics import recall_score

from src.datasets import make_synthetic_benchmark
from src.processor_benchmark import (
    BenchmarkSettings,
    BASIS_GATES,
    PROCESSOR_FILES,
    ProcessorSpec,
    load_trained_weights,
    run_benchmark,
)
from src.vqc import build_vqc_circuit


def write_parameter_fixture(path: Path) -> None:
    """Make a small schema-valid parameter file without training in tests."""

    dataset = make_synthetic_benchmark()
    _, _, weights = build_vqc_circuit()
    path.write_text(
        json.dumps(
            {
                "architecture": {"num_qubits": 4},
                "dataset": {
                    "source": dataset.source,
                    "train_samples": len(dataset.y_train),
                    "test_samples": len(dataset.y_test),
                    "features": dataset.X_test.shape[1],
                    "generation_random_state": dataset.generation_random_state,
                    "split_random_state": dataset.random_state,
                    "test_size": dataset.test_size,
                },
                "trained_parameters": [
                    {"name": str(parameter), "value_radians": float(value)}
                    for parameter, value in zip(
                        weights, np.linspace(-0.5, 0.5, len(weights)), strict=True
                    )
                ],
            }
        ),
        encoding="utf-8",
    )


class ProcessorConfigurationTests(unittest.TestCase):
    def test_maps_are_explicitly_custom_and_reference_requires_attribution(self):
        a, b = (ProcessorSpec.from_json(path) for path in PROCESSOR_FILES)
        self.assertEqual((a.num_qubits, b.num_qubits), (5, 9))
        for spec in (a, b):
            self.assertEqual(spec.coupling_map_source, "custom")
            self.assertIsNone(spec.reference_url)
            self.assertEqual({tuple(reversed(edge)) for edge in spec.coupling_edges}, set(spec.coupling_edges))
            with self.assertRaises(ValueError):
                replace(spec, coupling_map_source="reference").validate()

    def test_saved_weights_require_matching_dataset_and_parameter_order(self):
        dataset = make_synthetic_benchmark()
        _, _, parameter_names = build_vqc_circuit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            write_parameter_fixture(path)
            weights, digest = load_trained_weights(path, dataset, parameter_names)
            self.assertEqual(weights.shape, (16,))
            self.assertEqual(len(digest), 64)

            altered_dataset = replace(dataset, generation_random_state=999)
            with self.assertRaisesRegex(ValueError, "dataset mismatch"):
                load_trained_weights(path, altered_dataset, parameter_names)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["trained_parameters"][0]["name"] = "wrong"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameter names"):
                load_trained_weights(path, dataset, parameter_names)

    def test_transpiled_cx_gates_follow_each_custom_map(self):
        circuit, _, _ = build_vqc_circuit()
        for path in PROCESSOR_FILES:
            spec = ProcessorSpec.from_json(path)
            compiled = transpile(
                circuit,
                coupling_map=CouplingMap(list(spec.coupling_edges)),
                basis_gates=list(BASIS_GATES),
                optimization_level=1,
                seed_transpiler=42,
            )
            self.assertEqual(compiled.num_qubits, spec.num_qubits)
            allowed = set(spec.coupling_edges)
            for operation in compiled.data:
                if operation.operation.name == "cx":
                    edge = tuple(compiled.find_bit(qubit).index for qubit in operation.qubits)
                    self.assertIn(edge, allowed)


class ProcessorSimulationTests(unittest.TestCase):
    def test_seeded_ideal_noisy_results_and_routing_metrics(self):
        settings = BenchmarkSettings(shots=128, seed=17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            write_parameter_fixture(path)
            first = run_benchmark(
                parameters_path=path,
                output_path=Path(directory) / "first.json",
                settings=settings,
                sample_limit=6,
            )
            second = run_benchmark(
                parameters_path=path,
                output_path=Path(directory) / "second.json",
                settings=settings,
                sample_limit=6,
            )
            self.assertEqual(
                first,
                json.loads((Path(directory) / "first.json").read_text(encoding="utf-8")),
            )

        self.assertEqual(first, second)
        self.assertEqual(len(first["processors"]), 2)
        self.assertEqual(first["dataset"]["evaluated_samples"], 6)
        self.assertEqual(first["settings_shared_by_processors"]["shots_per_circuit"], 128)
        self.assertEqual(
            first["settings_shared_by_processors"]["noise_rates"],
            {
                "one_qubit_depolarizing": 0.001,
                "two_qubit_depolarizing": 0.01,
                "readout_flip": 0.02,
            },
        )
        reference = first["unconstrained_compile_baseline"]
        labels = first["dataset"]["true_labels"]
        for processor in first["processors"]:
            self.assertEqual(processor["coupling_map_source"], "custom")
            self.assertEqual(
                processor["routing_overhead_proxy"]["cx_delta_vs_unconstrained"],
                processor["cx_count"] - reference["cx_count"],
            )
            self.assertEqual(
                processor["routing_overhead_proxy"]["depth_delta_vs_unconstrained"],
                processor["depth"] - reference["depth"],
            )
            for mode in ("ideal", "noisy"):
                predictions = processor[mode]["predictions"]
                self.assertEqual(len(predictions), 6)
                self.assertEqual(set(predictions) - {0, 1}, set())
                self.assertEqual(
                    processor[mode]["recall_label_1"],
                    recall_score(labels, predictions, pos_label=1, zero_division=0),
                )
                self.assertEqual(len(processor[mode]["positive_probabilities"]), 6)
        self.assertGreater(
            first["processors"][0]["routing_overhead_proxy"]["cx_delta_vs_unconstrained"],
            first["processors"][1]["routing_overhead_proxy"]["cx_delta_vs_unconstrained"],
        )


if __name__ == "__main__":
    unittest.main()
