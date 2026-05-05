"""KS-based tabular drift detection using Alibi Detect."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from alibi_detect.cd import TabularDrift
from numpy.typing import NDArray

from src.config import FEATURE_COLUMNS


@dataclass(frozen=True)
class DriftResult:
    """Immutable snapshot of a drift test outcome."""

    is_drift: bool
    p_val_threshold: float
    feature_names: list[str]
    p_values: list[float]
    is_drift_per_feature: list[bool]
    n_reference: int
    n_test: int


def build_detector(
    reference_data: NDArray[np.floating],
    *,
    feature_names: list[str],
    p_val: float = 0.05,
) -> TabularDrift:
    """Fit a KS drift detector on *reference_data*."""
    return TabularDrift(
        x_ref=reference_data.astype(np.float32),
        p_val=p_val,
        categories_per_feature={},
    )


def run_drift_test(
    detector: TabularDrift,
    test_data: NDArray[np.floating],
    *,
    feature_names: list[str],
) -> DriftResult:
    """Run the detector on *test_data* and return a structured result."""
    pred = detector.predict(test_data.astype(np.float32))
    data = pred["data"]
    threshold = float(data["threshold"])
    p_values = [float(p) for p in data["p_val"]]
    return DriftResult(
        is_drift=bool(data["is_drift"]),
        p_val_threshold=threshold,
        feature_names=list(feature_names),
        p_values=p_values,
        is_drift_per_feature=[p < threshold for p in p_values],
        n_reference=int(detector.x_ref.shape[0]),
        n_test=int(test_data.shape[0]),
    )


def check_drift_from_features(
    features_path: Path | str,
    *,
    test_frac: float = 0.2,
    test_window: int = 30,
    p_val: float = 0.05,
) -> DriftResult:
    """End-to-end drift check: load features, split, detect."""
    import pandas as pd

    df = pd.read_csv(features_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"features CSV missing columns: {missing}")

    X = df[FEATURE_COLUMNS].values
    n = len(X)
    split_idx = int(n * (1.0 - test_frac))

    if split_idx < 50:
        raise ValueError(
            f"Need at least 50 reference rows, got {split_idx}"
        )
    if n - split_idx < test_window:
        raise ValueError(
            f"Test split has {n - split_idx} rows but test_window={test_window}"
        )

    reference = X[:split_idx]
    test = X[-test_window:]

    detector = build_detector(reference, feature_names=FEATURE_COLUMNS, p_val=p_val)
    return run_drift_test(detector, test, feature_names=FEATURE_COLUMNS)
