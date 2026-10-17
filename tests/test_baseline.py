"""Focused tests for the reproducible classical Q1 baseline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baseline import evaluate_logistic_regression, run_synthetic_baseline
from src.datasets import make_synthetic_benchmark


class LogisticRegressionBaselineTests(unittest.TestCase):
    def test_metrics_are_reproducible_and_cover_held_out_test_set(self):
        dataset = make_synthetic_benchmark()
        before_train = dataset.X_train.copy()
        before_test = dataset.X_test.copy()
        first = evaluate_logistic_regression(dataset)
        second = evaluate_logistic_regression(dataset)

        self.assertEqual(first, second)
        self.assertEqual(first.test_samples, 40)
        self.assertEqual(first.confusion_matrix_labels, [0, 1])
        self.assertEqual(first.confusion_matrix, [[17, 3], [2, 18]])
        self.assertEqual(first.accuracy, 0.875)
        self.assertEqual(first.abnormal_recall, 0.9)
        self.assertEqual(
            sum(sum(row) for row in first.confusion_matrix),
            len(dataset.y_test),
        )
        self.assertEqual(first.positive_label, 1)
        for metric in (first.accuracy, first.abnormal_recall, first.precision, first.f1):
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
        np.testing.assert_array_equal(dataset.X_train, before_train)
        np.testing.assert_array_equal(dataset.X_test, before_test)

    def test_run_saves_machine_readable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "baseline.json"
            result = run_synthetic_baseline(output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, result.to_dict())
        self.assertEqual(
            set(saved),
            {
                "accuracy",
                "abnormal_recall",
                "precision",
                "f1",
                "confusion_matrix",
                "confusion_matrix_labels",
                "test_samples",
                "positive_label",
                "model",
                "data",
                "sklearn_version",
            },
        )
        self.assertEqual(
            saved["data"]["split"],
            "stratified train/test split (80%/20%), random_state=42",
        )
        self.assertEqual(saved["data"]["generation_random_state"], 42)


if __name__ == "__main__":
    unittest.main()
