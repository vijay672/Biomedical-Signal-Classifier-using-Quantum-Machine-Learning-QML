"""Regression checks for saved-data statistics and the small compiler seed probe."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import nbformat

from src.judge_robustness import run as run_robustness
from src.judge_statistics import run as run_statistics, wilson_interval
from src.q1_submission import ROOT


class JudgeStatisticsTests(unittest.TestCase):
    def test_wilson_interval_boundaries(self):
        low, high = wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, .161, places=3)
        self.assertEqual(wilson_interval(20, 20)[1], 1.0)
        with self.assertRaises(ValueError):
            wilson_interval(21, 20)

    def test_saved_statistics_and_paired_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = run_statistics(Path(directory) / "review.json")
        self.assertTrue(payload["outer_test_rows_identical"])
        self.assertTrue(payload["validation_only_threshold_verified"])
        self.assertEqual((payload["inner_fit_rows"], payload["inner_validation_rows"]), (128, 32))
        cases = payload["cases"]
        self.assertEqual((cases["ideal_vqc"]["false_positives"],
                          cases["ideal_vqc"]["false_negatives"]), (3, 13))
        self.assertEqual(cases["separate_validation_model"]["specificity_label_0"], .25)
        paired = payload["paired_comparisons"]["processor_A_noisy_vs_processor_B_noisy"]
        self.assertEqual((paired["accuracy_discordance_first_only_correct"],
                          paired["accuracy_discordance_second_only_correct"]), (1, 3))
        self.assertEqual(paired["exact_two_sided_mcnemar_accuracy_p"], .625)

    def test_validation_label_tamper_is_rejected(self):
        from src import judge_statistics
        original = judge_statistics._load

        def corrupted(path):
            record = original(path)
            if path.name == "q1_recall_calibration.json":
                record["threshold_search"]["validation_labels"][0] ^= 1
            return record

        with tempfile.TemporaryDirectory() as directory, patch.object(
            judge_statistics, "_load", side_effect=corrupted
        ):
            with self.assertRaisesRegex(ValueError, "validation labels"):
                judge_statistics.run(Path(directory) / "invalid.json")

    def test_seed_probe_retains_frozen_seed_42_and_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = run_robustness(Path(directory) / "seeds.json")
        self.assertEqual(payload["seeds"], [7, 42, 99])
        self.assertEqual(len(payload["rows"]), 6)
        a = [r for r in payload["rows"] if r["processor"] == "Processor A"]
        b = [r for r in payload["rows"] if r["processor"] == "Processor B"]
        self.assertEqual({r["cx_count"] for r in a}, {24})
        self.assertEqual({r["cx_count"] for r in b}, {8})
        self.assertEqual({r["explicit_swap_count_after_sabre_layout"] for r in a}, {6})
        self.assertEqual({r["explicit_swap_count_after_sabre_layout"] for r in b}, {0})
        self.assertEqual(next(r for r in a if r["seed_transpiler"] == 42)["gate_only_depth"], 35)

    def test_saved_figures_expose_measured_labels(self):
        for name in ("judge_model_comparison", "recall_analysis", "recall_uncertainty", "layout_ablation"):
            ElementTree.parse(ROOT / "results" / "figures" / f"{name}.svg")
        comparison = (ROOT / "results/figures/judge_model_comparison.svg").read_text()
        recall = (ROOT / "results/figures/recall_analysis.svg").read_text()
        uncertainty = (ROOT / "results/figures/recall_uncertainty.svg").read_text()
        layout = (ROOT / "results/figures/layout_ablation.svg").read_text()
        self.assertIn("<!-- 0.875 -->", comparison)
        self.assertIn("<!-- 0.800 -->", recall)
        self.assertIn("<!-- 0.800 -->", uncertainty)
        for value in ("24", "22", "14", "26"):
            self.assertIn(f"<!-- {value} -->", layout)

    def test_notebook_primary_plot_uses_synthetic_label_language(self):
        notebook = nbformat.read(ROOT / "main.ipynb", as_version=4)
        nbformat.validate(notebook)
        sources = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn('"judge_model_comparison", "confusion_matrices"', sources)
        self.assertIn("Small-sample uncertainty and compiler seeds", sources)


if __name__ == "__main__":
    unittest.main()
