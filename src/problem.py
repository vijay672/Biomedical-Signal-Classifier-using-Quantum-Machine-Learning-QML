"""Q1 benchmark definition and verifiable data/preprocessing identity."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from typing import Any

import numpy as np

from src.datasets import PreparedDataset, make_provisional_q1_benchmark


IDENTITY_VERSION = 1


def _digest_array(array: np.ndarray, dtype: str) -> str:
    normalized = np.ascontiguousarray(array, dtype=dtype)
    descriptor = json.dumps({"shape": list(normalized.shape), "dtype": dtype}, sort_keys=True)
    return hashlib.sha256(descriptor.encode() + normalized.tobytes()).hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def dataset_identity(dataset: PreparedDataset) -> dict[str, Any]:
    """Fingerprint split contents and the fitted scaler, not just split seeds."""

    arrays = {
        name: _digest_array(getattr(dataset, name), "<i8" if name.startswith("y_") else "<f8")
        for name in ("raw_X_train", "raw_X_test", "X_train", "X_test", "y_train", "y_test")
    }
    scaler = dataset.scaler
    preprocessing = {
        "type": "MinMaxScaler",
        "feature_range": list(scaler.feature_range),
        "clip": bool(scaler.clip),
        "n_samples_seen": int(scaler.n_samples_seen_),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    }
    definition = {
        "source": dataset.source,
        "generation_parameters": dict(dataset.generation_parameters or {}),
        "split": {"method": "stratified 80/20", "test_size": dataset.test_size,
                  "random_state": dataset.random_state},
        "feature_names": list(dataset.feature_names),
        "scikit_learn_version": version("scikit-learn"),
    }
    return {
        "identity_version": IDENTITY_VERSION,
        "definition": definition,
        "array_sha256": arrays,
        "split_sha256": _digest_json(arrays),
        "preprocessing": preprocessing,
        "preprocessing_sha256": _digest_json(preprocessing),
        "dataset_sha256": _digest_json({"definition": definition, "arrays": arrays,
                                       "preprocessing": preprocessing}),
    }
