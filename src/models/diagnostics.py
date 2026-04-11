"""On-the-fly model diagnostics for the dashboard comparison view.

We recompute confusion matrices, ROC curves, and feature importances live
from the model + features CSV rather than storing them as MLflow artifacts.
This keeps the MLflow store lean and guarantees the dashboard always shows
numbers consistent with the checked-out features data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelDiagnostics:
    accuracy: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    roc_auc: Optional[float]
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]


def evaluate_on_split(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series | np.ndarray
) -> ModelDiagnostics:
    """Compute predictions, confusion matrix and ROC curve for one model."""
    y_test_arr = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))

    y_proba: Optional[np.ndarray]
    try:
        y_proba = np.asarray(model.predict_proba(X_test))[:, 1]
    except Exception:
        y_proba = None

    cm = confusion_matrix(y_test_arr, y_pred, labels=[0, 1])

    if y_proba is not None and len(np.unique(y_test_arr)) == 2:
        fpr, tpr, _ = roc_curve(y_test_arr, y_proba)
        roc_auc: Optional[float] = float(roc_auc_score(y_test_arr, y_proba))
    else:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        roc_auc = None

    return ModelDiagnostics(
        accuracy=float(accuracy_score(y_test_arr, y_pred)),
        confusion_matrix=cm,
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        y_pred=y_pred,
        y_proba=y_proba,
    )


def _unwrap_estimator(model: Any) -> Any:
    """Return the final estimator in a Pipeline, or the model itself."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def feature_importance(
    model: Any, feature_names: Sequence[str]
) -> Optional[pd.DataFrame]:
    """Return a sorted DataFrame of feature importances, or None if unavailable.

    Supports any estimator exposing ``feature_importances_`` (tree-based).
    Pipelines are unwrapped to their last step. Returns ``None`` for
    estimators that don't expose this attribute (e.g. plain LogisticRegression).
    """
    estimator = _unwrap_estimator(model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    importances = np.asarray(importances)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"feature_importances_ length {len(importances)} "
            f"!= feature_names length {len(feature_names)}"
        )
    return (
        pd.DataFrame(
            {"feature": list(feature_names), "importance": importances}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
