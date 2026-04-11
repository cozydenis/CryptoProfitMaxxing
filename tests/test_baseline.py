"""Tests for baseline model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.baseline import (
    SUPPORTED_MODELS,
    build_model,
    chronological_split,
    evaluate,
    load_features,
)


def _synthetic_features(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    # construct a signal so logreg can actually learn something
    signal = data["return_1d"] + 0.5 * data["rsi_14"]
    data[TARGET_COLUMN] = (signal > np.median(signal)).astype(int)
    return pd.DataFrame(data)


def test_chronological_split_preserves_order():
    df = _synthetic_features(n=100)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    split = chronological_split(X, y, test_frac=0.2)
    assert len(split.X_train) == 80
    assert len(split.X_test) == 20
    assert len(split.y_train) == 80
    assert len(split.y_test) == 20


def test_chronological_split_no_shuffle():
    """Train indices must come strictly before test indices in original order."""
    df = _synthetic_features(n=50)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    X["_orig_idx"] = np.arange(len(X))
    split = chronological_split(X, y, test_frac=0.2)
    assert split.X_train["_orig_idx"].max() < split.X_test["_orig_idx"].min()


def test_chronological_split_raises_on_bad_frac():
    df = _synthetic_features(n=100)
    with pytest.raises(ValueError):
        chronological_split(df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_frac=0.0)
    with pytest.raises(ValueError):
        chronological_split(df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_frac=1.0)


def test_chronological_split_raises_on_tiny_input():
    df = _synthetic_features(n=5)
    with pytest.raises(ValueError, match="at least 10"):
        chronological_split(df[FEATURE_COLUMNS], df[TARGET_COLUMN])


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_and_fit(model_name):
    df = _synthetic_features(n=150)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    split = chronological_split(X, y, test_frac=0.2)
    pipe = build_model(model_name)
    pipe.fit(split.X_train, split.y_train)
    metrics = evaluate(pipe, split.X_test, split.y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    assert metrics["n_test"] == len(split.y_test)


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("xgboost")


def test_load_features_round_trip(tmp_path):
    df = _synthetic_features(n=50)
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    X, y = load_features(path)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 50
    assert y.dtype == int or y.dtype == "int64"


def test_load_features_raises_on_missing_column(tmp_path):
    df = _synthetic_features(n=50).drop(columns=["rsi_14"])
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_features(path)
