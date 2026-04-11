"""MLflow query helpers for the dashboard comparison view.

These helpers are deliberately thin and pure so the dashboard logic that
consumes them can be unit-tested against a mocked ``MlflowClient``. The
module never imports Streamlit — caching is the dashboard's responsibility.

Feature ordering invariant: callers must feed loaded models the same
``FEATURE_COLUMNS`` ordering used by ``train.py``. The single source of truth
is ``src.config.FEATURE_COLUMNS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import pandas as pd
from mlflow.sklearn import load_model as _mlflow_load_model

COMPARISON_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
)


@dataclass(frozen=True)
class RunSummary:
    """Immutable snapshot of the fields we care about from an MLflow Run."""

    run_id: str
    model: str
    metrics: Mapping[str, float]
    params: Mapping[str, str]
    start_time: int  # ms since epoch
    experiment_id: str


def list_runs_by_model(
    client: Any, experiment_name: str
) -> dict[str, list[RunSummary]]:
    """Group all runs in ``experiment_name`` by their ``params.model`` value.

    Runs missing the ``model`` param are skipped. Each group is ordered most
    recent first. Returns an empty dict when the experiment does not exist or
    contains no runs.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
    )
    grouped: dict[str, list[RunSummary]] = {}
    for run in runs:
        params = dict(run.data.params or {})
        model = params.get("model")
        if not model:
            continue
        summary = RunSummary(
            run_id=run.info.run_id,
            model=model,
            metrics=dict(run.data.metrics or {}),
            params=params,
            start_time=int(run.info.start_time),
            experiment_id=run.info.experiment_id,
        )
        grouped.setdefault(model, []).append(summary)
    return grouped


def best_run_per_model(
    runs_by_model: Mapping[str, Iterable[RunSummary]]
) -> dict[str, RunSummary]:
    """Pick the winner for each model family.

    Tiebreaker chain: ``roc_auc`` desc, then ``accuracy`` desc, then
    ``start_time`` desc. Models with no runs are dropped from the result.
    """
    NEG_INF = float("-inf")

    def sort_key(r: RunSummary) -> tuple[float, float, int]:
        return (
            r.metrics.get("roc_auc", NEG_INF),
            r.metrics.get("accuracy", NEG_INF),
            r.start_time,
        )

    best: dict[str, RunSummary] = {}
    for model, runs in runs_by_model.items():
        runs_list = list(runs)
        if not runs_list:
            continue
        best[model] = max(runs_list, key=sort_key)
    return best


def metrics_dataframe(best_runs: Mapping[str, RunSummary]) -> pd.DataFrame:
    """One row per model, columns = COMPARISON_METRICS + run_id + start_time."""
    rows: list[dict[str, Any]] = []
    for model, run in best_runs.items():
        row: dict[str, Any] = {"model": model}
        for metric in COMPARISON_METRICS:
            row[metric] = run.metrics.get(metric)
        row["run_id"] = run.run_id
        row["start_time"] = run.start_time
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["model", *COMPARISON_METRICS, "run_id", "start_time"]
        )
    return pd.DataFrame(rows)


def runs_dataframe(runs: Iterable[RunSummary]) -> pd.DataFrame:
    """Flat dataframe of runs (for the 'all runs' expander)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {"run_id": run.run_id, "start_time": run.start_time}
        for metric in COMPARISON_METRICS:
            row[metric] = run.metrics.get(metric)
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["run_id", "start_time", *COMPARISON_METRICS]
        )
    return pd.DataFrame(rows)


def load_model_for_run(
    run_id: str,
    model_name: str,
    models_dir: Path,
) -> tuple[Any, str]:
    """Load the sklearn model for a run, with joblib fallback.

    Returns ``(model, source)`` where ``source`` is one of ``"mlflow"``,
    ``"joblib-fallback"``, or ``"missing"``. On ``"missing"`` the first tuple
    element is ``None``. Never raises — the dashboard stays demo-robust.
    """
    try:
        model = _mlflow_load_model(f"runs:/{run_id}/model")
        return model, "mlflow"
    except Exception:
        pass

    joblib_path = models_dir / f"baseline_{model_name}.pkl"
    if joblib_path.exists():
        try:
            return joblib.load(joblib_path), "joblib-fallback"
        except Exception:
            pass

    return None, "missing"
