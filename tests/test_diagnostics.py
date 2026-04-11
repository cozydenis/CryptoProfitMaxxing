"""Tests for src.models.diagnostics — uses real sklearn estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.baseline import build_model, chronological_split
from src.models.diagnostics import (
    ModelDiagnostics,
    evaluate_on_split,
    feature_importance,
)


def _synthetic_features(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    signal = data["return_1d"] + 0.5 * data["rsi_14"]
    data[TARGET_COLUMN] = (signal > np.median(signal)).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def fitted_rf_split():
    df = _synthetic_features(n=150)
    split = chronological_split(
        df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_frac=0.2
    )
    pipe = build_model("rf")
    pipe.fit(split.X_train, split.y_train)
    return pipe, split


@pytest.fixture
def fitted_logreg_split():
    df = _synthetic_features(n=150)
    split = chronological_split(
        df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_frac=0.2
    )
    pipe = build_model("logreg")
    pipe.fit(split.X_train, split.y_train)
    return pipe, split


class TestEvaluateOnSplit:
    def test_shape_and_types(self, fitted_rf_split):
        pipe, split = fitted_rf_split
        diag = evaluate_on_split(pipe, split.X_test, split.y_test)
        assert isinstance(diag, ModelDiagnostics)
        assert diag.confusion_matrix.shape == (2, 2)
        assert diag.y_pred.shape == (len(split.y_test),)
        assert diag.y_proba is not None
        assert diag.y_proba.shape == (len(split.y_test),)
        assert 0.0 <= diag.accuracy <= 1.0
        assert diag.roc_auc is not None
        assert 0.0 <= diag.roc_auc <= 1.0
        assert len(diag.fpr) == len(diag.tpr)
        assert diag.fpr[0] == 0.0
        assert diag.fpr[-1] == 1.0

    def test_handles_missing_proba(self):
        class NoProba:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        X = pd.DataFrame({c: [0.0, 1.0, 2.0] for c in FEATURE_COLUMNS})
        y = pd.Series([0, 1, 0])
        diag = evaluate_on_split(NoProba(), X, y)
        assert diag.y_proba is None
        assert diag.roc_auc is None
        # fallback diagonal
        assert np.array_equal(diag.fpr, np.array([0.0, 1.0]))
        assert np.array_equal(diag.tpr, np.array([0.0, 1.0]))

    def test_confusion_matrix_matches_sklearn(self, fitted_logreg_split):
        from sklearn.metrics import confusion_matrix as sk_cm

        pipe, split = fitted_logreg_split
        diag = evaluate_on_split(pipe, split.X_test, split.y_test)
        expected = sk_cm(split.y_test, diag.y_pred, labels=[0, 1])
        assert np.array_equal(diag.confusion_matrix, expected)


class TestFeatureImportance:
    def test_rf_returns_sorted_frame(self, fitted_rf_split):
        pipe, _ = fitted_rf_split
        fi = feature_importance(pipe, FEATURE_COLUMNS)
        assert fi is not None
        assert list(fi.columns) == ["feature", "importance"]
        assert len(fi) == len(FEATURE_COLUMNS)
        # sorted descending
        assert fi["importance"].is_monotonic_decreasing
        # importances sum to approximately 1 for RF
        assert fi["importance"].sum() == pytest.approx(1.0, rel=1e-6)

    def test_logreg_returns_none(self, fitted_logreg_split):
        pipe, _ = fitted_logreg_split
        assert feature_importance(pipe, FEATURE_COLUMNS) is None

    def test_plain_estimator_unwrapping(self):
        # Non-pipeline tree estimator
        from sklearn.tree import DecisionTreeClassifier

        X = np.random.default_rng(0).normal(size=(50, len(FEATURE_COLUMNS)))
        y = np.random.default_rng(0).integers(0, 2, size=50)
        tree = DecisionTreeClassifier(random_state=0).fit(X, y)
        fi = feature_importance(tree, FEATURE_COLUMNS)
        assert fi is not None
        assert len(fi) == len(FEATURE_COLUMNS)

    def test_mismatched_feature_names_raises(self, fitted_rf_split):
        pipe, _ = fitted_rf_split
        with pytest.raises(ValueError, match="length"):
            feature_importance(pipe, ["only_one"])
