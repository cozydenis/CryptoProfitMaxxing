"""Hyperparameter tuning with Ray Tune + MLflow tracking.

Usage:
    python tune.py --model logreg
    python tune.py --model rf --num-samples 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TUNE_MAX_CONCURRENT,
    DEFAULT_TUNE_NUM_SAMPLES,
)
from src.models.baseline import chronological_split, load_features
from src.tuning.runner import run_tuning


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parent / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f).get("tune", {})


def main() -> int:
    defaults = _load_params()
    parser = argparse.ArgumentParser(description="Tune crypto model hyperparameters with Ray Tune")
    parser.add_argument(
        "--model", choices=["logreg", "rf"], required=True,
    )
    parser.add_argument("--features", default=defaults.get("features_path", "data/processed/features.csv"))
    parser.add_argument("--test-frac", type=float, default=defaults.get("test_frac", 0.2))
    parser.add_argument("--random-state", type=int, default=defaults.get("random_state", 42))
    parser.add_argument(
        "--num-samples", type=int,
        default=defaults.get("num_samples", DEFAULT_TUNE_NUM_SAMPLES),
    )
    parser.add_argument(
        "--max-concurrent", type=int,
        default=defaults.get("max_concurrent_trials", DEFAULT_TUNE_MAX_CONCURRENT),
    )
    parser.add_argument(
        "--experiment",
        default=defaults.get("experiment_name", DEFAULT_EXPERIMENT_NAME),
    )
    args = parser.parse_args()

    X, y = load_features(args.features)
    split = chronological_split(X, y, test_frac=args.test_frac)

    print(f"\n=== Ray Tune: {args.model} ({args.num_samples} trials) ===\n")

    summary = run_tuning(
        model_name=args.model,
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        random_state=args.random_state,
        experiment_name=args.experiment,
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent,
    )

    print(f"\nTrials completed: {summary['num_trials']}")
    print(f"Best config: {json.dumps(summary['best_config'], indent=2)}")
    print(f"Best roc_auc: {summary['best_metrics'].get('roc_auc', 'N/A')}")
    print(f"Best accuracy: {summary['best_metrics'].get('accuracy', 'N/A')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
