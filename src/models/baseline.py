"""Baseline classifiers (logreg / random forest) with chronological split.

Time-series data must NOT be shuffled: the canonical ML-on-crypto mistake is
using ``sklearn.train_test_split(shuffle=True)`` which leaks future information
into the training set. We implement an explicit chronological split that keeps
the first ``1 - test_frac`` of rows for training and the last ``test_frac`` for
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, TARGET_COLUMN

SUPPORTED_MODELS = ("logreg", "rf")


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_features(path: str | "pd.PathLike[str]") -> Tuple[pd.DataFrame, pd.Series]:
    """Load features CSV and return (X, y) with FEATURE_COLUMNS / TARGET_COLUMN."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"features CSV missing columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2
) -> SplitData:
    """Split X, y in time order — first (1 - test_frac) train, last test_frac test.

    Rows are assumed to already be sorted chronologically (the ingestion and
    featurize stages enforce this).
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    n = len(X)
    if n < 10:
        raise ValueError("need at least 10 rows to split")
    split_idx = int(n * (1.0 - test_frac))
    return SplitData(
        X_train=X.iloc[:split_idx].reset_index(drop=True),
        X_test=X.iloc[split_idx:].reset_index(drop=True),
        y_train=y.iloc[:split_idx].reset_index(drop=True),
        y_test=y.iloc[split_idx:].reset_index(drop=True),
    )


def build_model(name: str, random_state: int = 42) -> Pipeline:
    """Return a fresh, unfit sklearn pipeline for the named model."""
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model '{name}'; supported: {SUPPORTED_MODELS}")
    if name == "logreg":
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        estimator = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=random_state, n_jobs=-1
        )
    return Pipeline(
        steps=[("scaler", StandardScaler()), ("model", estimator)]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Return a metrics dict suitable for logging to MLflow / JSON."""
    preds = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc: float | None = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None
    cm = confusion_matrix(y_test, preds)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "n_test": int(len(y_test)),
        "positive_rate_test": float(np.mean(y_test)),
        "confusion_matrix": cm.tolist(),
    }
    if auc is not None:
        metrics["roc_auc"] = auc
    return metrics
