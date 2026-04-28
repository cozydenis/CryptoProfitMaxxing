"""Search space definitions and pipeline factory for Ray Tune."""

from __future__ import annotations

from typing import Any

from ray import tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SUPPORTED = ("logreg", "rf")


def get_search_space(model_name: str) -> dict[str, Any]:
    """Return a Ray Tune search space dict for the given model."""
    if model_name not in _SUPPORTED:
        raise ValueError(f"unknown model '{model_name}'; supported: {_SUPPORTED}")

    if model_name == "logreg":
        return {
            "C": tune.loguniform(1e-3, 100),
            "penalty": tune.choice(["l1", "l2"]),
            "solver": tune.choice(["saga"]),  # saga supports both l1 and l2
        }

    return {
        "n_estimators": tune.choice([50, 100, 200, 300]),
        "max_depth": tune.choice([3, 5, 8, 12, None]),
        "min_samples_split": tune.choice([2, 5, 10]),
        "min_samples_leaf": tune.choice([1, 2, 4]),
        "max_features": tune.choice(["sqrt", "log2", None]),
    }


def build_pipeline_from_config(
    model_name: str,
    config: dict[str, Any],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Build an sklearn Pipeline from a sampled config dict."""
    if model_name not in _SUPPORTED:
        raise ValueError(f"unknown model '{model_name}'; supported: {_SUPPORTED}")

    if model_name == "logreg":
        estimator = LogisticRegression(
            C=config.get("C", 1.0),
            penalty=config.get("penalty", "l2"),
            solver=config.get("solver", "saga"),
            max_iter=1000,
            random_state=random_state,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", None),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            max_features=config.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=-1,
        )

    return Pipeline(steps=[("scaler", StandardScaler()), ("model", estimator)])
