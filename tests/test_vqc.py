"""Shape and parameter tests for the four-qubit Q1 VQC."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datasets import make_synthetic_benchmark
from src.vqc import (
    NUM_QUBITS,
    NUM_WEIGHTS,
    VQCConfig,
    build_vqc_circuit,
    build_vqc_components,
    fit_vqc,
    save_vqc_parameters,
)


class VQCCircuitTests(unittest.TestCase):
    def test_four_qubit_circuit_has_expected_parameter_and_layer_counts(self):
        circuit, inputs, weights = build_vqc_circuit()

        self.assertEqual(circuit.num_qubits, NUM_QUBITS)
        self.assertEqual(len(inputs), NUM_QUBITS)
        self.assertEqual(len(weights), NUM_WEIGHTS)
        self.assertEqual(circuit.num_parameters, NUM_QUBITS + NUM_WEIGHTS)
        self.assertEqual(circuit.count_ops()["ry"], NUM_QUBITS + NUM_QUBITS * 2)
        self.assertEqual(circuit.count_ops()["rz"], NUM_QUBITS * 2)
        self.assertEqual(circuit.count_ops()["cx"], NUM_QUBITS * 2)
        self.assertEqual(circuit.count_ops()["measure"], NUM_QUBITS)

    def test_qnn_outputs_two_probabilities_per_input(self):
        first = build_vqc_components(VQCConfig(random_state=7, maxiter=18))
        second = build_vqc_components(VQCConfig(random_state=7, maxiter=18))
        dataset = make_synthetic_benchmark()

        np.testing.assert_array_equal(first.initial_weights, second.initial_weights)
        output = first.qnn.forward(dataset.X_test[:3], first.initial_weights)
        self.assertEqual(output.shape, (3, 2))
        np.testing.assert_allclose(output.sum(axis=1), np.ones(3), atol=1e-8)

    def test_optimizer_budget_is_bounded(self):
        with self.assertRaises(ValueError):
            VQCConfig(maxiter=NUM_WEIGHTS + 1)


class VQCTrainingTests(unittest.TestCase):
    def test_fitted_predictions_shapes_and_saved_parameters(self):
        dataset = make_synthetic_benchmark()
        result = fit_vqc(dataset, VQCConfig(random_state=11, maxiter=18))

        self.assertEqual(result.trained_weights.shape, (NUM_WEIGHTS,))
        self.assertEqual(result.test_probabilities.shape, (40, 2))
        self.assertEqual(result.test_predictions.shape, (40,))
        self.assertTrue(np.isfinite(result.test_probabilities).all())
        self.assertTrue(np.isin(result.test_predictions, (0, 1)).all())
        self.assertLessEqual(result.optimizer_evaluations, 18)

        with tempfile.TemporaryDirectory() as directory:
            destination = save_vqc_parameters(
                result, dataset, Path(directory) / "vqc-parameters.json"
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["trained_parameters"]), NUM_WEIGHTS)
        self.assertEqual(payload["architecture"]["num_qubits"], NUM_QUBITS)
        self.assertEqual(payload["training"]["optimizer"], "COBYLA")
        self.assertEqual(payload["dataset"]["test_samples"], 40)
        self.assertEqual(payload["held_out"]["probability_shape"], [40, 2])


if __name__ == "__main__":
    unittest.main()
