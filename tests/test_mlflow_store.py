"""Tests for src.mlflow_store — uses mocked MlflowClient."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from src.config import TUNE_TAG_KEY
from src.mlflow_store import (
    COMPARISON_METRICS,
    RunSummary,
    best_run_per_model,
    list_runs_by_model,
    load_model_for_run,
    metrics_dataframe,
    runs_dataframe,
    tuning_summary,
)


def _make_run(
    run_id: str,
    model: str,
    metrics: dict[str, float],
    start_time: int,
    experiment_id: str = "1",
    tags: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Build a minimal object that mimics an MLflow Run's attribute surface."""
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id, experiment_id=experiment_id, start_time=start_time
        ),
        data=SimpleNamespace(
            params={"model": model, "test_frac": "0.2"},
            metrics=metrics,
            tags=tags or {},
        ),
    )


def _client_with_runs(runs: list[SimpleNamespace], experiment_id: str = "1") -> MagicMock:
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id=experiment_id, name="crypto-baseline"
    )
    client.search_runs.return_value = runs
    return client


class TestListRunsByModel:
    def test_groups_by_param_model(self):
        runs = [
            _make_run("r1", "logreg", {"accuracy": 0.5, "roc_auc": 0.48}, 1000),
            _make_run("r2", "logreg", {"accuracy": 0.55, "roc_auc": 0.52}, 2000),
            _make_run("r3", "rf", {"accuracy": 0.6, "roc_auc": 0.58}, 3000),
        ]
        client = _client_with_runs(runs)
        grouped = list_runs_by_model(client, "crypto-baseline")
        assert set(grouped.keys()) == {"logreg", "rf"}
        assert len(grouped["logreg"]) == 2
        assert len(grouped["rf"]) == 1

    def test_skips_runs_without_model_param(self):
        run = _make_run("r1", "logreg", {"accuracy": 0.5}, 1000)
        run.data.params = {"other": "value"}  # no model param
        client = _client_with_runs([run])
        grouped = list_runs_by_model(client, "crypto-baseline")
        assert grouped == {}

    def test_empty_experiment_returns_empty_dict(self):
        client = _client_with_runs([])
        grouped = list_runs_by_model(client, "crypto-baseline")
        assert grouped == {}

    def test_missing_experiment_returns_empty_dict(self):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        grouped = list_runs_by_model(client, "does-not-exist")
        assert grouped == {}
        client.search_runs.assert_not_called()

    def test_summary_carries_expected_fields(self):
        runs = [
            _make_run(
                "rX",
                "logreg",
                {"accuracy": 0.5, "roc_auc": 0.48},
                1234,
                experiment_id="42",
            )
        ]
        client = _client_with_runs(runs, experiment_id="42")
        grouped = list_runs_by_model(client, "crypto-baseline")
        summary = grouped["logreg"][0]
        assert isinstance(summary, RunSummary)
        assert summary.run_id == "rX"
        assert summary.model == "logreg"
        assert summary.metrics["accuracy"] == 0.5
        assert summary.params["model"] == "logreg"
        assert summary.start_time == 1234
        assert summary.experiment_id == "42"


class TestBestRunPerModel:
    def test_prefers_roc_auc(self):
        r_high_acc_low_auc = RunSummary(
            "r1", "logreg", {"accuracy": 0.9, "roc_auc": 0.50}, {}, 1000, "1"
        )
        r_low_acc_high_auc = RunSummary(
            "r2", "logreg", {"accuracy": 0.6, "roc_auc": 0.85}, {}, 2000, "1"
        )
        grouped = {"logreg": [r_high_acc_low_auc, r_low_acc_high_auc]}
        best = best_run_per_model(grouped)
        assert best["logreg"].run_id == "r2"

    def test_falls_back_to_accuracy_when_auc_missing(self):
        r_low = RunSummary("r1", "logreg", {"accuracy": 0.5}, {}, 1000, "1")
        r_high = RunSummary("r2", "logreg", {"accuracy": 0.7}, {}, 2000, "1")
        best = best_run_per_model({"logreg": [r_low, r_high]})
        assert best["logreg"].run_id == "r2"

    def test_falls_back_to_start_time_on_full_tie(self):
        r_old = RunSummary(
            "r1", "rf", {"accuracy": 0.5, "roc_auc": 0.6}, {}, 1000, "1"
        )
        r_new = RunSummary(
            "r2", "rf", {"accuracy": 0.5, "roc_auc": 0.6}, {}, 2000, "1"
        )
        best = best_run_per_model({"rf": [r_old, r_new]})
        assert best["rf"].run_id == "r2"

    def test_handles_multiple_models(self):
        grouped = {
            "logreg": [
                RunSummary("a", "logreg", {"roc_auc": 0.7}, {}, 1, "1"),
                RunSummary("b", "logreg", {"roc_auc": 0.8}, {}, 2, "1"),
            ],
            "rf": [RunSummary("c", "rf", {"roc_auc": 0.9}, {}, 3, "1")],
        }
        best = best_run_per_model(grouped)
        assert best["logreg"].run_id == "b"
        assert best["rf"].run_id == "c"

    def test_empty_input_returns_empty_dict(self):
        assert best_run_per_model({}) == {}

    def test_drops_models_with_no_runs(self):
        grouped = {"logreg": [], "rf": [RunSummary("c", "rf", {}, {}, 1, "1")]}
        best = best_run_per_model(grouped)
        assert "logreg" not in best
        assert "rf" in best


class TestMetricsDataframe:
    def test_one_row_per_model(self):
        best = {
            "logreg": RunSummary(
                "r1", "logreg", {"accuracy": 0.6, "f1": 0.55}, {}, 100, "1"
            ),
            "rf": RunSummary(
                "r2", "rf", {"accuracy": 0.7, "f1": 0.65}, {}, 200, "1"
            ),
        }
        df = metrics_dataframe(best)
        assert len(df) == 2
        assert set(df["model"]) == {"logreg", "rf"}
        assert df.set_index("model").loc["logreg", "accuracy"] == 0.6
        assert df.set_index("model").loc["rf", "f1"] == 0.65

    def test_missing_metrics_become_nan(self):
        best = {
            "logreg": RunSummary("r1", "logreg", {"accuracy": 0.6}, {}, 100, "1")
        }
        df = metrics_dataframe(best)
        assert df.loc[0, "accuracy"] == 0.6
        assert pd.isna(df.loc[0, "roc_auc"])

    def test_empty_input_returns_empty_frame_with_schema(self):
        df = metrics_dataframe({})
        assert list(df.columns) == [
            "model",
            *COMPARISON_METRICS,
            "run_id",
            "start_time",
        ]
        assert len(df) == 0


class TestRunsDataframe:
    def test_flat_layout(self):
        runs = [
            RunSummary("r1", "logreg", {"accuracy": 0.5}, {}, 100, "1"),
            RunSummary("r2", "logreg", {"accuracy": 0.6}, {}, 200, "1"),
        ]
        df = runs_dataframe(runs)
        assert list(df["run_id"]) == ["r1", "r2"]
        assert "accuracy" in df.columns

    def test_empty(self):
        df = runs_dataframe([])
        assert len(df) == 0
        assert "accuracy" in df.columns


class TestLoadModelForRun:
    def test_returns_mlflow_when_successful(self, tmp_path):
        fake_model = object()
        with patch(
            "src.mlflow_store._mlflow_load_model", return_value=fake_model
        ) as loader:
            model, source = load_model_for_run("rid", "logreg", tmp_path)
        loader.assert_called_once_with("runs:/rid/model")
        assert model is fake_model
        assert source == "mlflow"

    def test_falls_back_to_joblib(self, tmp_path):
        import joblib

        sentinel = {"kind": "fallback"}
        joblib_path = tmp_path / "baseline_logreg.pkl"
        joblib.dump(sentinel, joblib_path)

        with patch(
            "src.mlflow_store._mlflow_load_model",
            side_effect=RuntimeError("no artifact"),
        ):
            model, source = load_model_for_run("rid", "logreg", tmp_path)

        assert model == sentinel
        assert source == "joblib-fallback"

    def test_returns_missing_when_both_fail(self, tmp_path):
        with patch(
            "src.mlflow_store._mlflow_load_model",
            side_effect=RuntimeError("nope"),
        ):
            model, source = load_model_for_run("rid", "logreg", tmp_path)
        assert model is None
        assert source == "missing"


class TestTuningSource:
    """RunSummary.tuning_source field and tuning_summary helper."""

    def test_default_tuning_source_is_none(self):
        r = RunSummary("r1", "logreg", {}, {}, 1000, "1")
        assert r.tuning_source is None

    def test_tuning_source_set_explicitly(self):
        r = RunSummary("r1", "logreg", {}, {}, 1000, "1", tuning_source="ray-tune")
        assert r.tuning_source == "ray-tune"

    def test_list_runs_populates_tuning_source(self):
        runs = [
            _make_run("r1", "logreg", {"accuracy": 0.5}, 1000, tags={TUNE_TAG_KEY: "ray-tune"}),
            _make_run("r2", "logreg", {"accuracy": 0.6}, 2000),
        ]
        client = _client_with_runs(runs)
        grouped = list_runs_by_model(client, "crypto-baseline")
        sources = [r.tuning_source for r in grouped["logreg"]]
        assert "ray-tune" in sources
        assert None in sources

    def test_tuning_summary_counts(self):
        runs = {
            "logreg": [
                RunSummary("r1", "logreg", {"roc_auc": 0.6}, {}, 1, "1", tuning_source="ray-tune"),
                RunSummary("r2", "logreg", {"roc_auc": 0.7}, {}, 2, "1", tuning_source="ray-tune"),
                RunSummary("r3", "logreg", {"roc_auc": 0.5}, {}, 3, "1"),
            ],
        }
        summary = tuning_summary(runs)
        assert summary["logreg"]["total_tuned_runs"] == 2
        assert summary["logreg"]["total_manual_runs"] == 1

    def test_tuning_summary_empty_when_no_runs(self):
        assert tuning_summary({}) == {}
