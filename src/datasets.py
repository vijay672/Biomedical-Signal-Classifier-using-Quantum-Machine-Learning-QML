"""Small data-loading and train-only preprocessing utilities for Q1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


FloatArray = NDArray[np.float64]
LabelArray = NDArray[np.int64]


@dataclass(frozen=True)
class PreparedDataset:
    """Train/test data with a scaler fitted only on the training partition.

    ``X_train`` and ``X_test`` are quantum inputs in [0, pi]. Test values are
    clipped to that range if they fall outside the training feature bounds.
    Raw partitions are retained to make preprocessing auditable.
    """

    X_train: FloatArray
    X_test: FloatArray
    y_train: LabelArray
    y_test: LabelArray
    raw_X_train: FloatArray
    raw_X_test: FloatArray
    scaler: MinMaxScaler
    feature_names: tuple[str, ...]
    source: str
    test_size: float
    random_state: int
    generation_random_state: int | None
    generation_parameters: Mapping[str, int] | None = None


@dataclass(frozen=True)
class MITBIHCSVConfig:
    """Settings for a local MIT-BIH-derived feature table.

    The adapter reads a user-provided CSV of already extracted numeric
    features. It does not download MIT-BIH records or assume a feature schema.
    ``label_map`` maps source labels to binary class IDs; rows with labels not
    in the mapping are ignored (useful for excluding unrelated arrhythmias).
    """

    csv_path: Path | str
    feature_columns: tuple[str, ...]
    label_column: str
    label_map: Mapping[str, int]


def prepare_dataset(
    features: Sequence[Sequence[float]] | NDArray[np.generic],
    labels: Sequence[int] | NDArray[np.generic],
    *,
    feature_names: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    source: str = "provided_features",
    generation_random_state: int | None = None,
    generation_parameters: Mapping[str, int] | None = None,
) -> PreparedDataset:
    """Stratify a binary dataset and scale features using training data only."""

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D array")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("labels must be a 1D array with one label per row")
    if not np.isfinite(X).all():
        raise ValueError("features must contain only finite numeric values")
    if not np.isin(y, (0, 1)).all() or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    names = tuple(feature_names) if feature_names is not None else tuple(
        f"feature_{i}" for i in range(X.shape[1])
    )
    if len(names) != X.shape[1]:
        raise ValueError("feature_names must have one name per feature")

    raw_train, raw_test, y_train, y_test = train_test_split(
        X,
        y.astype(np.int64, copy=False),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)), clip=True)
    X_train = scaler.fit_transform(raw_train)
    X_test = scaler.transform(raw_test)

    return PreparedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        raw_X_train=raw_train,
        raw_X_test=raw_test,
        scaler=scaler,
        feature_names=names,
        source=source,
        test_size=test_size,
        random_state=random_state,
        generation_random_state=generation_random_state,
        generation_parameters=generation_parameters,
    )


def make_synthetic_benchmark() -> PreparedDataset:
    """Create the requested reproducible 200-sample, four-feature fixture."""

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        random_state=42,
    )
    return prepare_dataset(
        X,
        y,
        feature_names=tuple(f"feature_{i}" for i in range(4)),
        source="make_classification",
        generation_random_state=42,
        generation_parameters={"n_samples": 200, "n_features": 4, "random_state": 42},
    )


def make_provisional_q1_benchmark() -> PreparedDataset:
    """Master Guide Q1 fixture with an explicit, unconfirmed redundant-feature choice."""

    parameters = {
        "n_samples": 200,
        "n_features": 4,
        "n_informative": 3,
        "n_redundant": 0,  # Engineering assumption: organizer example omits this and fails.
        "random_state": 42,
    }
    X, y = make_classification(**parameters)
    return prepare_dataset(
        X,
        y,
        feature_names=tuple(f"feature_{i}" for i in range(4)),
        source="make_classification_q1_provisional_n_redundant_0",
        generation_random_state=42,
        generation_parameters=parameters,
    )


def load_mitbih_feature_csv(config: MITBIHCSVConfig) -> tuple[FloatArray, LabelArray]:
    """Load selected binary classes from a configurable local feature CSV.

    CSV records outside ``config.label_map`` are skipped. The returned arrays
    are raw and should be passed to :func:`prepare_dataset` for a fresh,
    training-only split and scaler fit.
    """

    path = Path(config.csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MIT-BIH feature CSV not found: {path}")
    if not config.feature_columns:
        raise ValueError("feature_columns must name at least one CSV column")
    if config.label_column in config.feature_columns:
        raise ValueError("label_column must not also be a feature column")
    if set(config.label_map.values()) != {0, 1}:
        raise ValueError("label_map must map selected labels to both 0 and 1")

    rows: list[list[float]] = []
    labels: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or ())
        required = set(config.feature_columns) | {config.label_column}
        missing = sorted(required - headers)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            source_label = (row.get(config.label_column) or "").strip()
            if source_label not in config.label_map:
                continue
            try:
                values = [float(row[column]) for column in config.feature_columns]
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid feature value on CSV row {row_number}") from error
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite feature value on CSV row {row_number}")
            rows.append(values)
            labels.append(config.label_map[source_label])

    if not rows:
        raise ValueError("CSV contains no rows matching label_map")
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)
