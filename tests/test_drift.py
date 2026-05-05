"""Tests for Alibi Detect drift monitoring — synthetic data, no network."""

from __future__ import annotations

import numpy as np
import pytest
from src.drift.detector import DriftResult, build_detector, run_drift_test


class TestBuildDetector:
    def test_returns_detector(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((50, 5))
        detector = build_detector(ref, feature_names=[f"f{i}" for i in range(5)])
        assert hasattr(detector, "predict")


class TestRunDriftTest:
    def test_no_drift_same_distribution(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 5))
        test = rng.standard_normal((30, 5))
        names = [f"f{i}" for i in range(5)]
        detector = build_detector(ref, feature_names=names)
        result = run_drift_test(detector, test, feature_names=names)
        assert isinstance(result, DriftResult)
        assert result.is_drift is False

    def test_detects_shifted_distribution(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 5))
        test = rng.standard_normal((30, 5)) + 5.0  # massive shift
        names = [f"f{i}" for i in range(5)]
        detector = build_detector(ref, feature_names=names)
        result = run_drift_test(detector, test, feature_names=names)
        assert result.is_drift is True
        assert all(p < 0.05 for p in result.p_values)

    def test_partial_shift(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 4))
        test = rng.standard_normal((30, 4))
        # Shift only the first feature
        test[:, 0] += 10.0
        names = [f"f{i}" for i in range(4)]
        detector = build_detector(ref, feature_names=names)
        result = run_drift_test(detector, test, feature_names=names)
        assert result.is_drift_per_feature[0] is True
        # Other features should mostly not drift
        assert not all(result.is_drift_per_feature[1:])


class TestDriftResult:
    def test_is_frozen(self) -> None:
        result = DriftResult(
            is_drift=False,
            p_val_threshold=0.05,
            feature_names=["a"],
            p_values=[0.5],
            is_drift_per_feature=[False],
            n_reference=100,
            n_test=30,
        )
        with pytest.raises(AttributeError):
            result.is_drift = True  # type: ignore[misc]

    def test_fields(self) -> None:
        result = DriftResult(
            is_drift=True,
            p_val_threshold=0.05,
            feature_names=["a", "b"],
            p_values=[0.01, 0.80],
            is_drift_per_feature=[True, False],
            n_reference=80,
            n_test=20,
        )
        assert result.n_reference == 80
        assert len(result.p_values) == 2
