"""Tests for the requested Q1 data preparation layer."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datasets import (
    MITBIHCSVConfig,
    load_mitbih_feature_csv,
    make_synthetic_benchmark,
    prepare_dataset,
)


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_reproducible_stratified_binary_split_and_shapes(self):
        first = make_synthetic_benchmark()
        second = make_synthetic_benchmark()

        np.testing.assert_array_equal(first.raw_X_train, second.raw_X_train)
        np.testing.assert_array_equal(first.raw_X_test, second.raw_X_test)
        np.testing.assert_array_equal(first.X_train, second.X_train)
        np.testing.assert_array_equal(first.y_train, second.y_train)
        np.testing.assert_array_equal(first.y_test, second.y_test)
        self.assertEqual(first.raw_X_train.shape, (160, 4))
        self.assertEqual(first.raw_X_test.shape, (40, 4))
        self.assertEqual(first.X_train.shape, (160, 4))
        self.assertEqual(first.X_test.shape, (40, 4))
        self.assertEqual(set(first.y_train), {0, 1})
        self.assertEqual(set(first.y_test), {0, 1})
        np.testing.assert_array_equal(np.bincount(first.y_train), [80, 80])
        np.testing.assert_array_equal(np.bincount(first.y_test), [20, 20])
        self.assertEqual(first.feature_names, tuple(f"feature_{i}" for i in range(4)))

    def test_quantum_inputs_are_bounded(self):
        result = make_synthetic_benchmark()
        self.assertTrue(np.isfinite(result.X_train).all())
        self.assertTrue(np.isfinite(result.X_test).all())
        self.assertGreaterEqual(float(result.X_train.min()), 0.0)
        self.assertLessEqual(float(result.X_train.max()), np.pi)
        self.assertGreaterEqual(float(result.X_test.min()), 0.0)
        self.assertLessEqual(float(result.X_test.max()), np.pi)

    def test_preprocessor_fits_training_rows_only(self):
        X = np.arange(400, dtype=float).reshape(100, 4)
        y = np.array([0, 1] * 50)
        baseline = prepare_dataset(X, y, random_state=19)

        changed_test = X.copy()
        # Use the actual deterministic split indices to alter held-out values.
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=19, stratify=y
        )
        changed_test[test_idx] += 1_000_000
        changed = prepare_dataset(changed_test, y, random_state=19)

        np.testing.assert_array_equal(baseline.raw_X_train, changed.raw_X_train)
        np.testing.assert_array_equal(baseline.X_train, changed.X_train)
        np.testing.assert_array_equal(baseline.scaler.data_min_, changed.scaler.data_min_)
        np.testing.assert_array_equal(baseline.scaler.data_max_, changed.scaler.data_max_)
        self.assertEqual(len(train_idx), len(baseline.raw_X_train))
        self.assertEqual(len(test_idx), len(baseline.raw_X_test))


class MITBIHAdapterTests(unittest.TestCase):
    def test_configurable_local_feature_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mitbih_features.csv"
            path.write_text(
                "rr_interval,amplitude,beat_class\n"
                "0.8,1.2,N\n"
                "0.6,2.1,V\n"
                "0.9,1.1,N\n"
                "0.5,2.4,V\n"
                "0.7,9.9,OTHER\n",
                encoding="utf-8",
            )
            config = MITBIHCSVConfig(
                csv_path=path,
                feature_columns=("rr_interval", "amplitude"),
                label_column="beat_class",
                label_map={"N": 0, "V": 1},
            )
            X, y = load_mitbih_feature_csv(config)

        np.testing.assert_array_equal(X, [[0.8, 1.2], [0.6, 2.1], [0.9, 1.1], [0.5, 2.4]])
        np.testing.assert_array_equal(y, [0, 1, 0, 1])

    def test_missing_local_data_is_reported_without_download_attempt(self):
        config = MITBIHCSVConfig(
            csv_path="/path/that/does/not/exist.csv",
            feature_columns=("f1",),
            label_column="label",
            label_map={"N": 0, "V": 1},
        )
        with self.assertRaises(FileNotFoundError):
            load_mitbih_feature_csv(config)

    def test_label_column_cannot_be_used_as_a_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            path.write_text("value,label\n0.1,0\n0.2,1\n", encoding="utf-8")
            config = MITBIHCSVConfig(
                csv_path=path,
                feature_columns=("value", "label"),
                label_column="label",
                label_map={"0": 0, "1": 1},
            )
            with self.assertRaisesRegex(ValueError, "label_column"):
                load_mitbih_feature_csv(config)


if __name__ == "__main__":
    unittest.main()
