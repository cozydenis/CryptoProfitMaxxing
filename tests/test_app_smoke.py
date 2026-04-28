"""Streamlit AppTest smoke tests for app.py.

Covers three scenarios via monkeypatching ``app.list_runs_by_model`` and
``app.load_model_for_run``, plus a temp ``PROCESSED_DATA_DIR`` with a
synthetic features CSV so tests don't depend on `dvc repro` having run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.mlflow_store import RunSummary
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeBaseModel:
    """Minimal sklearn-duck-typed model for AppTest scenarios."""

    def __init__(self, prediction: int, probability: float):
        self._pred = prediction
        self._proba = probability

    def predict(self, X):  # noqa: N803
        n = len(X)
        return np.full(n, self._pred, dtype=int)

    def predict_proba(self, X):  # noqa: N803
        n = len(X)
        p = np.full(n, self._proba)
        return np.stack([1.0 - p, p], axis=1)


class FakeRF(FakeBaseModel):
    def __init__(self, prediction: int, probability: float, n_features: int):
        super().__init__(prediction, probability)
        rng = np.random.default_rng(0)
        self.feature_importances_ = rng.dirichlet(np.ones(n_features))


def _make_features_csv(path: Path, n: int = 60) -> None:
    """Write a synthetic features CSV with all FEATURE_COLUMNS + target + timestamp + close."""
    rng = np.random.default_rng(7)
    now = datetime.now(tz=timezone.utc)
    timestamps = pd.date_range(end=now, periods=n, freq="D", tz="UTC")
    data: dict = {
        "timestamp": timestamps,
        "close": 30_000.0 + rng.normal(0, 500, size=n).cumsum(),
        "volume": rng.uniform(8e8, 1.2e9, size=n),
    }
    for col in FEATURE_COLUMNS:
        data[col] = rng.normal(size=n)
    data[TARGET_COLUMN] = rng.integers(0, 2, size=n)
    if "sma_20" in data:
        data["sma_20"] = pd.Series(data["close"]).rolling(5, min_periods=1).mean().values
    if "sma_50" in data:
        data["sma_50"] = pd.Series(data["close"]).rolling(10, min_periods=1).mean().values
    pd.DataFrame(data).to_csv(path, index=False)


def _run_summary(
    run_id: str, model: str, start_time: int = 1_700_000_000_000
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        model=model,
        metrics={
            "accuracy": 0.55,
            "precision": 0.52,
            "recall": 0.58,
            "f1": 0.55,
            "roc_auc": 0.54 if model == "rf" else 0.49,
        },
        params={"model": model, "test_frac": "0.2"},
        start_time=start_time,
        experiment_id="1",
    )


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Clear all Streamlit caches before and after each test.

    ``AppTest.from_file`` re-executes ``app.py`` as a fresh module each call,
    so patches must target the source modules (``src.mlflow_store``,
    ``src.config``) — patches on ``app.*`` don't survive script re-execution.
    Caches still persist across AppTest sessions at the process level, so we
    must clear them explicitly.
    """
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def features_dir(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    _make_features_csv(processed / "features.csv")
    # Patch at the source module so app.py's `from src.config import ...`
    # picks it up on re-execution.
    import src.config

    monkeypatch.setattr(src.config, "PROCESSED_DATA_DIR", processed)
    return processed


@pytest.fixture
def patch_mlflow(monkeypatch):
    """Returns a setter you call with the runs dict and the model-loader impl."""
    import src.mlflow_store

    def _apply(
        runs_by_model: dict[str, list[RunSummary]],
        loader_impl=lambda rid, name, models_dir: (None, "missing"),
    ):
        monkeypatch.setattr(
            src.mlflow_store,
            "list_runs_by_model",
            lambda client, exp: runs_by_model,
        )
        monkeypatch.setattr(
            src.mlflow_store, "load_model_for_run", loader_impl
        )

    return _apply


def _texts(at):
    """Flatten every renderable string node's value into one big string for contains checks."""
    parts: list[str] = []
    for attr in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "info",
        "warning",
        "success",
        "error",
        "metric",
    ):
        for node in getattr(at, attr, []):
            value = getattr(node, "value", None)
            if value is not None:
                parts.append(str(value))
            label = getattr(node, "label", None)
            if label is not None:
                parts.append(str(label))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Happy path — two models in MLflow
# ---------------------------------------------------------------------------


def test_happy_path_renders_full_comparison(features_dir, patch_mlflow):
    runs = {
        "logreg": [_run_summary("r-lr-1", "logreg")],
        "rf": [_run_summary("r-rf-1", "rf")],
    }

    def loader(run_id, model_name, models_dir):
        n_features = len(FEATURE_COLUMNS)
        if model_name == "rf":
            return FakeRF(prediction=1, probability=0.62, n_features=n_features), "mlflow"
        return FakeBaseModel(prediction=0, probability=0.44), "mlflow"

    patch_mlflow(runs, loader)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]

    text = _texts(at)
    assert "CryptoProfitMaxxing" in text
    assert "Model comparison" in text
    assert "Best run per model" in text
    assert "Next-day prediction" in text
    assert "Metric comparison" in text
    assert "Diagnostics on held-out test split" in text
    assert "Models disagree" in text  # rf=UP, logreg=DOWN
    assert "RF" in text and "LOGREG" in text


# ---------------------------------------------------------------------------
# Degraded — only one model
# ---------------------------------------------------------------------------


def test_degraded_single_model(features_dir, patch_mlflow):
    runs = {"rf": [_run_summary("r-rf-1", "rf")]}

    def loader(run_id, model_name, models_dir):
        return (
            FakeRF(prediction=1, probability=0.6, n_features=len(FEATURE_COLUMNS)),
            "mlflow",
        )

    patch_mlflow(runs, loader)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]

    text = _texts(at)
    assert "Only one model available" in text
    assert "Metric comparison" not in text  # full comparison skipped
    assert "Diagnostics on held-out test split" not in text


# ---------------------------------------------------------------------------
# Empty — no runs at all
# ---------------------------------------------------------------------------


def test_empty_no_runs(features_dir, patch_mlflow):
    patch_mlflow({})

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]

    text = _texts(at)
    assert "No runs found" in text
    assert "Best run per model" not in text


# ---------------------------------------------------------------------------
# Features missing entirely
# ---------------------------------------------------------------------------


def test_no_features_csv(tmp_path, monkeypatch, patch_mlflow):
    import src.config

    empty_dir = tmp_path / "empty_processed"
    empty_dir.mkdir()
    monkeypatch.setattr(src.config, "PROCESSED_DATA_DIR", empty_dir)
    patch_mlflow({"rf": [_run_summary("r-rf-1", "rf")]})

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]

    text = _texts(at)
    assert "No features CSV found" in text


# ---------------------------------------------------------------------------
# Disagreement banner logic: models agreeing should show success/warning
# ---------------------------------------------------------------------------


def test_both_models_agree_up(features_dir, patch_mlflow):
    runs = {
        "logreg": [_run_summary("r-lr-1", "logreg")],
        "rf": [_run_summary("r-rf-1", "rf")],
    }

    def loader(run_id, model_name, models_dir):
        n = len(FEATURE_COLUMNS)
        if model_name == "rf":
            return FakeRF(prediction=1, probability=0.7, n_features=n), "mlflow"
        return FakeBaseModel(prediction=1, probability=0.65), "mlflow"

    patch_mlflow(runs, loader)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]

    text = _texts(at)
    assert "All models agree: **UP**" in text
