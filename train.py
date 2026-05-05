"""Train a baseline model with MLflow tracking + model registry.

Usage:
    python train.py --model logreg
    python train.py --model rf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MLRUNS_DIR,
    MODELS_DIR,
)
from src.models.baseline import (
    build_model,
    chronological_split,
    evaluate,
    load_features,
)
from src.models.lstm import LSTMClassifier


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parent / "params.yaml"
    with params_path.open() as f:
        all_params = yaml.safe_load(f)
    return all_params["train"], all_params.get("lstm", {})


def main() -> int:
    defaults, lstm_defaults = _load_params()
    parser = argparse.ArgumentParser(description="Train a baseline crypto model")
    parser.add_argument(
        "--model", choices=["logreg", "rf", "lstm"], default=defaults["model"]
    )
    parser.add_argument("--features", default=defaults["features_path"])
    parser.add_argument("--test-frac", type=float, default=defaults["test_frac"])
    parser.add_argument(
        "--random-state", type=int, default=defaults["random_state"]
    )
    parser.add_argument(
        "--experiment", default=defaults.get("experiment_name", DEFAULT_EXPERIMENT_NAME)
    )
    parser.add_argument(
        "--registered-model",
        default=defaults.get("registered_model_name", DEFAULT_REGISTERED_MODEL_NAME),
    )
    parser.add_argument("--no-register", action="store_true", help="Skip model registry")
    args = parser.parse_args()

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment)

    X, y = load_features(args.features)
    split = chronological_split(X, y, test_frac=args.test_frac)

    if args.model == "lstm":
        pipeline = LSTMClassifier(
            seq_len=lstm_defaults.get("seq_len", 10),
            hidden_size=lstm_defaults.get("hidden_size", 64),
            num_layers=lstm_defaults.get("num_layers", 1),
            dropout=lstm_defaults.get("dropout", 0.0),
            lr=lstm_defaults.get("lr", 1e-3),
            epochs=lstm_defaults.get("epochs", 50),
            batch_size=lstm_defaults.get("batch_size", 32),
            random_state=args.random_state,
        )
    else:
        pipeline = build_model(args.model, random_state=args.random_state)

    with mlflow.start_run(run_name=args.model) as run:
        mlflow.log_params(
            {
                "model": args.model,
                "test_frac": args.test_frac,
                "random_state": args.random_state,
                "n_features": X.shape[1],
                "n_train": len(split.y_train),
                "n_test": len(split.y_test),
                "positive_rate_train": float(split.y_train.mean()),
            }
        )
        pipeline.fit(split.X_train, split.y_train)
        metrics = evaluate(pipeline, split.X_test, split.y_test)

        scalar_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        mlflow.log_metrics(scalar_metrics)

        registered_name = None if args.no_register else args.registered_model
        if args.model == "lstm":
            mlflow.log_params(pipeline.get_params())
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=registered_name,
        )

        artifact_path = MODELS_DIR / f"baseline_{args.model}.pkl"
        joblib.dump(pipeline, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="local_joblib")

        metrics_path = MODELS_DIR / "metrics.json"
        metrics_payload = {
            "model": args.model,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            **metrics,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        mlflow.log_artifact(str(metrics_path))

        print(f"\n=== {args.model} ===")
        print(json.dumps(scalar_metrics, indent=2))
        print(f"Run ID: {run.info.run_id}")
        print(f"Model saved: {artifact_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
