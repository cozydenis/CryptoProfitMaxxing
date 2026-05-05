"""Ray Tune orchestration — objective function and tuning runner."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from ray import tune
from ray.tune import ResultGrid, TuneConfig

from src.config import (
    MLFLOW_TRACKING_URI,
    MLRUNS_DIR,
    MODELS_DIR,
    TUNE_TAG_KEY,
    TUNE_TAG_VALUE,
)
from src.models.baseline import evaluate
from src.tuning.search_spaces import build_pipeline_from_config, get_search_space


def run_single_trial(
    config: dict[str, Any],
    *,
    model_name: str,
    random_state: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train one config and return metrics. Used both by Ray Tune and tests."""
    pipeline = build_pipeline_from_config(model_name, config, random_state=random_state)
    try:
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
    except ValueError:
        # Degenerate split (e.g. single class) — return zero metrics
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if "roc_auc" not in metrics:
        metrics["roc_auc"] = 0.0
    return metrics


def _objective(config: dict[str, Any]) -> None:
    """Ray Tune trainable — trains, logs to MLflow, reports metrics."""
    model_name: str = config.pop("_model_name")
    random_state: int = config.pop("_random_state")
    experiment_name: str = config.pop("_experiment_name")
    X_train: pd.DataFrame = config.pop("_X_train")
    y_train: pd.Series = config.pop("_y_train")
    X_test: pd.DataFrame = config.pop("_X_test")
    y_test: pd.Series = config.pop("_y_test")

    metrics = run_single_trial(
        config,
        model_name=model_name,
        random_state=random_state,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    pipeline = build_pipeline_from_config(model_name, config, random_state=random_state)
    pipeline.fit(X_train, y_train)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    with mlflow.start_run(run_name=f"tune-{model_name}"):
        mlflow.set_tag(TUNE_TAG_KEY, TUNE_TAG_VALUE)
        mlflow.log_params({"model": model_name, **config})
        mlflow.log_metrics(scalar_metrics)
        try:
            mlflow.sklearn.log_model(pipeline, "model")
        except Exception:
            pass  # don't crash the trial if artifact logging fails

    tune.report({"roc_auc": metrics.get("roc_auc", 0.0), "accuracy": metrics["accuracy"]})


def run_tuning(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    experiment_name: str = "crypto-baseline",
    num_samples: int = 20,
    max_concurrent_trials: int = 2,
) -> dict[str, Any]:
    """Run Ray Tune search and return a summary dict."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    search_space = get_search_space(model_name)

    # Inject context into config so _objective can read it
    search_space["_model_name"] = model_name
    search_space["_random_state"] = random_state
    search_space["_experiment_name"] = experiment_name
    search_space["_X_train"] = X_train
    search_space["_y_train"] = y_train
    search_space["_X_test"] = X_test
    search_space["_y_test"] = y_test

    tuner = tune.Tuner(
        _objective,
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            metric="roc_auc",
            mode="max",
        ),
    )

    results: ResultGrid = tuner.fit()
    best = results.get_best_result(metric="roc_auc", mode="max")

    best_config = {
        k: v for k, v in (best.config or {}).items() if not k.startswith("_")
    }

    return {
        "best_config": best_config,
        "best_metrics": best.metrics or {},
        "num_trials": len(results),
    }
