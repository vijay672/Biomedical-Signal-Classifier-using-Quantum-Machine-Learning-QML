"""Official-path regression checks for dataset and strict model identity."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.artifacts import build_model_artifact, load_verified_model, save_model_artifact
from src.datasets import make_provisional_q1_benchmark, make_synthetic_benchmark
from src.problem import dataset_identity
from src.q1_submission import PRESCRIBED_NOISE, smoke_test
from src.processors import ProcessorSpec
from src.routing import COMPILER_SETTINGS, sabre_swap_trace
from src.vqc import VQCConfig, build_vqc_circuit


class ProvisionalBenchmarkTests(unittest.TestCase):
    def test_corrected_fixture_is_distinct_and_training_fitted(self):
        first = make_provisional_q1_benchmark()
        second = make_provisional_q1_benchmark()
        self.assertEqual(first.generation_parameters["n_informative"], 3)
        self.assertEqual(first.generation_parameters["n_redundant"], 0)
        self.assertEqual(first.X_train.shape, (160, 4))
        self.assertEqual(first.X_test.shape, (40, 4))
        np.testing.assert_array_equal(first.raw_X_train, second.raw_X_train)
        np.testing.assert_allclose(first.scaler.data_min_, first.raw_X_train.min(axis=0))
        np.testing.assert_allclose(first.scaler.data_max_, first.raw_X_train.max(axis=0))
        self.assertNotEqual(dataset_identity(first)["dataset_sha256"],
                            dataset_identity(make_synthetic_benchmark())["dataset_sha256"])

    def test_artifact_rejects_architecture_data_and_scaler_changes(self):
        dataset = make_provisional_q1_benchmark()
        config = VQCConfig()
        payload = build_model_artifact(
            dataset, np.linspace(-0.5, 0.5, 16), [], config=config,
            optimizer_evaluations=0, final_objective=0.0,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model_artifact(payload, path)
            self.assertEqual(load_verified_model(path, dataset)[0].shape, (16,))

            changed = json.loads(json.dumps(payload))
            changed["architecture"]["definition"]["readout"] = "first qubit"
            save_model_artifact(changed, path)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_verified_model(path, dataset)
            save_model_artifact(payload, path)

            raw_train = dataset.raw_X_train.copy()
            raw_train[0, 0] += 0.25
            with self.assertRaisesRegex(ValueError, "dataset"):
                load_verified_model(path, replace(dataset, raw_X_train=raw_train))

            scaler = dataset.scaler
            from copy import deepcopy
            changed_scaler = deepcopy(scaler)
            changed_scaler.data_min_[0] -= 0.25
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                load_verified_model(path, replace(dataset, scaler=changed_scaler))

            changed = json.loads(json.dumps(payload))
            changed["trained_parameters"][0]["name"] = "other"
            save_model_artifact(changed, path)
            with self.assertRaisesRegex(ValueError, "parameter"):
                load_verified_model(path, dataset)

            changed = json.loads(json.dumps(payload))
            changed["environment"]["numpy"] = "0.0.0"
            save_model_artifact(changed, path)
            with self.assertRaisesRegex(ValueError, "software-version"):
                load_verified_model(path, dataset)

    def test_official_compiler_and_noise_smoke(self):
        self.assertEqual(COMPILER_SETTINGS["layout_method"], "sabre")
        self.assertEqual(COMPILER_SETTINGS["routing_method"], "sabre")
        self.assertEqual(COMPILER_SETTINGS["optimization_level"], 1)
        self.assertEqual(COMPILER_SETTINGS["seed_transpiler"], 42)
        self.assertEqual(PRESCRIBED_NOISE.two_qubit_depolarizing, 0.005)
        self.assertEqual(PRESCRIBED_NOISE.readout_flip, 0.01)
        self.assertEqual(smoke_test()["cx_counts"], [24, 8])

    def test_processor_graphs_and_sabre_swap_trace(self):
        logical, _, _ = build_vqc_circuit()
        paths = (Path("processors/processor_A.json"), Path("processors/processor_B.json"))
        specs = [ProcessorSpec.from_json(path) for path in paths]
        self.assertEqual([spec.num_qubits for spec in specs], [5, 9])
        self.assertTrue(all(spec.coupling_map_source == "custom" for spec in specs))
        counts = [sabre_swap_trace(logical, spec)["verified_inserted_swap_count"] for spec in specs]
        self.assertEqual(counts, [6, 0])
