"""Tests for the Ray Tune runner — mocked Ray and MLflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.tuning.runner import run_single_trial


@pytest.fixture
def synthetic_split() -> dict:
    """Synthetic train/test data for testing the objective."""
    rng = np.random.default_rng(42)
    n_train, n_test, n_feat = 80, 20, 3
    cols = ["a", "b", "c"]
    return {
        "X_train": pd.DataFrame(rng.standard_normal((n_train, n_feat)), columns=cols),
        "y_train": pd.Series(rng.integers(0, 2, size=n_train)),
        "X_test": pd.DataFrame(rng.standard_normal((n_test, n_feat)), columns=cols),
        "y_test": pd.Series(rng.integers(0, 2, size=n_test)),
    }


class TestRunSingleTrial:
    """The objective function trains a model and returns metrics."""

    def test_returns_metrics_dict(self, synthetic_split: dict) -> None:
        config = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
        result = run_single_trial(
            config=config,
            model_name="logreg",
            random_state=42,
            **synthetic_split,
        )
        assert "accuracy" in result
        assert "roc_auc" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_rf_trial_returns_metrics(self, synthetic_split: dict) -> None:
        config = {
            "n_estimators": 50,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        result = run_single_trial(
            config=config,
            model_name="rf",
            random_state=42,
            **synthetic_split,
        )
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_handles_degenerate_single_class(self) -> None:
        """When all labels are the same class, roc_auc should be 0.0."""
        rng = np.random.default_rng(99)
        cols = ["a", "b", "c"]
        X_train = pd.DataFrame(rng.standard_normal((40, 3)), columns=cols)
        y_train = pd.Series(np.ones(40, dtype=int))  # all class 1
        X_test = pd.DataFrame(rng.standard_normal((10, 3)), columns=cols)
        y_test = pd.Series(np.ones(10, dtype=int))
        config = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
        result = run_single_trial(
            config=config,
            model_name="logreg",
            random_state=42,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        assert result["roc_auc"] == 0.0
