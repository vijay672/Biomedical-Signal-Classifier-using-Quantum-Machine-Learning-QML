"""Focused checks for the bounded, training-only VQC selection protocol."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datasets import make_synthetic_benchmark
from src.result_figures import render_result_figures
from src.vqc import VQCConfig, build_vqc_circuit, first_qubit_class
from src.vqc_experiments import make_inner_split, select_variant


class ControlledExperimentTests(unittest.TestCase):
    def test_inner_split_is_stratified_and_scaler_uses_fit_rows_only(self):
        dataset = make_synthetic_benchmark()
        split = make_inner_split(dataset)
        self.assertEqual((len(split.y_fit), len(split.y_validation)), (128, 32))
        self.assertEqual(np.bincount(split.y_validation).tolist(), [16, 16])
        self.assertFalse(set(split.fit_indices) & set(split.validation_indices))
        np.testing.assert_allclose(
            split.scaler.data_min_, dataset.raw_X_train[split.fit_indices].min(axis=0)
        )
        np.testing.assert_allclose(
            split.scaler.data_max_, dataset.raw_X_train[split.fit_indices].max(axis=0)
        )
        np.testing.assert_allclose(
            split.X_validation,
            split.scaler.transform(dataset.raw_X_train[split.validation_indices]),
        )

    def test_variant_parameterization_and_bounded_config(self):
        ring, _, ring_weights = build_vqc_circuit()
        line, _, line_weights = build_vqc_circuit("line")
        self.assertEqual((ring.count_ops()["cx"], line.count_ops()["cx"]), (8, 6))
        self.assertEqual(len(ring_weights), len(line_weights))
        self.assertEqual(first_qubit_class(0b0101), 1)
        self.assertEqual(first_qubit_class(0b0110), 0)
        with self.assertRaises(ValueError):
            VQCConfig(maxiter=101)

    def test_selection_uses_validation_recall_before_precision(self):
        def candidate(name, recall, precision):
            return {
                "variant": {"name": name},
                "validation": {
                    "recall_label_1": recall,
                    "precision_label_1": precision,
                    "f1_label_1": 0.5,
                },
                "compilation": {"processors": {
                    "A": {"routing_overhead_cx": 0},
                    "B": {"routing_overhead_cx": 0},
                }},
                "held_out": {"recall_label_1": 1.0 if name == "lower" else 0.0},
            }
        selected = select_variant([
            candidate("lower", 0.5, 1.0), candidate("higher", 0.75, 0.1)
        ])
        self.assertEqual(selected["variant"]["name"], "higher")

    def test_figures_reflect_recorded_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = {
                "baseline.json": {
                    "accuracy": 0.75, "abnormal_recall": 0.5,
                    "precision": 0.5, "f1": 0.5,
                },
                "experiment.json": {
                    "final_refit_and_held_out": {"metrics": {
                        "confusion_matrix": [[2, 0], [1, 1]], "sample_count": 4,
                        "recall_label_1": 0.5, "precision_label_1": 1.0,
                        "f1_label_1": 2 / 3,
                    }},
                    "candidates": [{
                        "variant": {"name": "ring_parity"},
                        "validation": {"recall_label_1": 0.5,
                                       "precision_label_1": 0.5, "f1_label_1": 0.5},
                    }],
                },
                "processor.json": {"processors": [{
                    "name": "Processor A", "num_qubits": 5,
                    "ideal": {"recall_label_1": 0.5},
                    "noisy": {"recall_label_1": 0.25},
                }]},
            }
            for name, payload in fixtures.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            paths = render_result_figures(
                baseline_path=root / "baseline.json",
                experiment_path=root / "experiment.json",
                processor_path=root / "processor.json",
                output_dir=root / "figures",
            )
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertIn("0.750", paths[0].read_text())
            self.assertIn("custom Processor A/B maps", paths[2].read_text())


if __name__ == "__main__":
    unittest.main()
