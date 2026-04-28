"""Tests for Ray Tune search space definitions and pipeline factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.tuning.search_spaces import (
    build_pipeline_from_config,
    get_search_space,
)


class TestGetSearchSpace:
    """Search space dicts must contain expected keys per model."""

    def test_logreg_has_expected_keys(self) -> None:
        space = get_search_space("logreg")
        assert "C" in space
        assert "penalty" in space

    def test_rf_has_expected_keys(self) -> None:
        space = get_search_space("rf")
        for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"):
            assert key in space, f"RF space missing key: {key}"

    def test_rejects_unknown_model(self) -> None:
        with pytest.raises(ValueError, match="unknown model"):
            get_search_space("xgboost")


class TestBuildPipelineFromConfig:
    """Pipeline factory must produce fitted-ready sklearn Pipelines."""

    def test_logreg_pipeline_structure(self) -> None:
        config = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
        pipe = build_pipeline_from_config("logreg", config, random_state=42)
        assert isinstance(pipe, Pipeline)
        assert isinstance(pipe.named_steps["scaler"], StandardScaler)
        estimator = pipe.named_steps["model"]
        assert estimator.C == 1.0
        assert estimator.penalty == "l2"

    def test_rf_pipeline_structure(self) -> None:
        config = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        pipe = build_pipeline_from_config("rf", config, random_state=42)
        assert isinstance(pipe, Pipeline)
        estimator = pipe.named_steps["model"]
        assert estimator.n_estimators == 100
        assert estimator.max_depth == 5

    def test_logreg_l1_forces_saga_solver(self) -> None:
        config = {"C": 0.5, "penalty": "l1", "solver": "saga"}
        pipe = build_pipeline_from_config("logreg", config, random_state=42)
        assert pipe.named_steps["model"].solver == "saga"

    def test_rejects_unknown_model(self) -> None:
        with pytest.raises(ValueError, match="unknown model"):
            build_pipeline_from_config("xgboost", {}, random_state=42)

    def test_pipeline_fits_and_predicts(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, size=50))
        config = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
        pipe = build_pipeline_from_config("logreg", config, random_state=42)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (50,)
        assert set(preds).issubset({0, 1})
