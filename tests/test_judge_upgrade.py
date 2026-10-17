"""Focused checks for the separate recall experiment and layout ablation."""

import hashlib
import json
import unittest

import numpy as np

from src.datasets import make_provisional_q1_benchmark
from src.judge_diagnostics import ABLATION_PATH, AUDIT_PATH
from src.metrics import classification_metrics
from src.q1_submission import BENCHMARK_PATH, MODEL_PATH, ROOT
from src.recall_calibration import MODEL_NAME, RESULT_NAME, choose_threshold


class JudgeUpgradeTests(unittest.TestCase):
    def test_threshold_selection_uses_validation_constraints(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([.8, .55, .6, .2])
        selected, grid = choose_threshold(
            labels, scores, thresholds=(.4, .6), min_precision=.6, min_f1=.6,
        )
        self.assertEqual(selected["threshold"], .4)
        self.assertEqual(selected["validation_metrics"]["recall_label_1"], 1.0)
        self.assertEqual(len(grid), 2)

    def test_threshold_fallback_is_explicit(self):
        selected, _ = choose_threshold(
            np.array([1, 0]), np.array([.2, .9]), min_precision=.8, min_f1=.8,
        )
        self.assertEqual(selected["threshold"], .5)
        self.assertIn("fallback", selected["status"])

    def test_inner_split_scaler_and_test_metrics(self):
        root = ROOT / "results" / "tables"
        result = json.loads((root / RESULT_NAME).read_text())
        model_path = root / MODEL_NAME
        model = json.loads(model_path.read_text())
        self.assertEqual(result["model_file_sha256"], hashlib.sha256(model_path.read_bytes()).hexdigest())
        fit = model["fit_indices_in_outer_training"]
        validation = model["validation_indices_in_outer_training"]
        self.assertEqual(len(fit), 128)
        self.assertEqual(len(validation), 32)
        self.assertEqual(set(fit) | set(validation), set(range(160)))
        self.assertFalse(set(fit) & set(validation))
        dataset = make_provisional_q1_benchmark()
        np.testing.assert_allclose(model["inner_scaler"]["data_min"],
                                   dataset.raw_X_train[fit].min(axis=0))
        np.testing.assert_allclose(model["inner_scaler"]["data_max"],
                                   dataset.raw_X_train[fit].max(axis=0))
        self.assertEqual(model["inner_scaler"]["n_samples_seen"], 128)
        selected = result["threshold_search"]["selection"]["threshold"]
        scores = np.array(result["held_out_test"]["positive_scores"])
        measured = classification_metrics(dataset.y_test, (scores > selected).astype(int))
        self.assertEqual(measured, result["held_out_test"]["metrics"])
        self.assertEqual(result["held_out_test"]["evaluated_once_after_selection"], True)

    def test_ablation_preserves_official_compilation(self):
        ablation = json.loads(ABLATION_PATH.read_text())
        benchmark = json.loads(BENCHMARK_PATH.read_text())
        self.assertEqual(len(ablation["rows"]), 4)
        for official in benchmark["processors"]:
            rows = [row for row in ablation["rows"] if row["processor"] == official["name"]]
            self.assertEqual({row["layout_method"] for row in rows}, {"sabre", "trivial"})
            for row in rows:
                self.assertEqual(row["compiler_settings"]["routing_method"], "sabre")
                self.assertEqual(row["compiler_settings"]["seed_transpiler"], 42)
                self.assertEqual(row["compiler_settings"]["optimization_level"], 1)
            self.assertEqual(next(row for row in rows if row["layout_method"] == "sabre")["cost"],
                             official["circuit_cost"])

    def test_original_artifact_hash_and_exact_score_replay(self):
        audit = json.loads(AUDIT_PATH.read_text())
        self.assertEqual(audit["model_file_sha256"], hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())
        for relative, expected in audit["original_artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        self.assertEqual(audit["exact_qnn_score_replay_max_absolute_error"], 0.0)
        self.assertEqual(set(audit["final_layer_rz_max_score_changes"]),
                         {"w[9]", "w[11]", "w[13]", "w[15]"})
        self.assertTrue(all(delta < 1e-12 for delta in audit["final_layer_rz_max_score_changes"].values()))


if __name__ == "__main__":
    unittest.main()
