"""Streamlit dashboard for CryptoProfitMaxxing.

Shows the BTC price chart, side-by-side model comparison (best run per model
from MLflow), next-day predictions with a disagreement banner, a metric bar
chart, ROC overlay, confusion matrices, Random Forest feature importance,
and an "all runs" audit expander. Refresh button in the sidebar invalidates
caches so new training runs can be picked up mid-demo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mlflow.tracking import MlflowClient

from src.config import (
    DEFAULT_EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)
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
from src.models.baseline import chronological_split
from src.models.diagnostics import (
    ModelDiagnostics,
    evaluate_on_split,
    feature_importance,
)

try:
    from src.drift.detector import DriftResult, check_drift_from_features

    _DRIFT_AVAILABLE = True
except ImportError:
    _DRIFT_AVAILABLE = False

STALE_FEATURES_HOURS = 48

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(
    page_title="CryptoProfitMaxxing",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _cached_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data(ttl=300, show_spinner=False)
def _cached_runs_by_model(
    experiment_name: str,
) -> dict[str, list[RunSummary]]:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return list_runs_by_model(client, experiment_name)


@st.cache_resource(show_spinner=False)
def _cached_model_for_run(run_id: str, model_name: str) -> tuple[Any, str]:
    return load_model_for_run(run_id, model_name, MODELS_DIR)


# ---------------------------------------------------------------------------
# Render helpers (pure — take data, return a chart or write to st)
# ---------------------------------------------------------------------------


def _render_sidebar(
    runs_by_model: dict[str, list[RunSummary]], df: pd.DataFrame
) -> None:
    st.sidebar.header("MLflow")
    st.sidebar.caption(f"Experiment: **{DEFAULT_EXPERIMENT_NAME}**")

    n_models = len(runs_by_model)
    n_runs = sum(len(v) for v in runs_by_model.values())
    st.sidebar.caption(f"{n_runs} runs across {n_models} models")

    if runs_by_model:
        latest_start = max(
            r.start_time for runs in runs_by_model.values() for r in runs
        )
        latest_dt = datetime.fromtimestamp(latest_start / 1000, tz=timezone.utc)
        st.sidebar.caption(f"Latest run: {latest_dt:%Y-%m-%d %H:%M UTC}")

    if st.sidebar.button("Refresh from MLflow", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if not df.empty:
        latest_candle = pd.to_datetime(df["timestamp"].iloc[-1])
        if latest_candle.tzinfo is None:
            latest_candle = latest_candle.tz_localize("UTC")
        age = datetime.now(tz=timezone.utc) - latest_candle.to_pydatetime()
        if age > timedelta(hours=STALE_FEATURES_HOURS):
            hours = int(age.total_seconds() // 3600)
            st.sidebar.warning(
                f":warning: Features are {hours}h old. Run `dvc repro` to refresh."
            )

    st.sidebar.divider()
    st.sidebar.caption(
        "CryptoProfitMaxxing — MLOps demo  \n"
        "CoinGecko + DVC + MLflow + Ray Tune + Streamlit"
    )


def _render_header(df: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest BTC close", f"${df['close'].iloc[-1]:,.2f}")
    col1.caption(
        f"as of {pd.to_datetime(df['timestamp'].iloc[-1]).date()}"
    )
    col2.metric("Feature rows", f"{len(df):,}")
    col3.metric("Feature columns", str(len(FEATURE_COLUMNS)))


def _price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            name="BTC close",
            line=dict(width=2),
        )
    )
    for col, style in (("sma_20", "dash"), ("sma_50", "dot")):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[col],
                    mode="lines",
                    name=col.upper().replace("_", " "),
                    line=dict(width=1, dash=style),
                )
            )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            x=0,
        ),
    )
    return fig


def _render_metrics_table(metrics_df: pd.DataFrame) -> None:
    display = metrics_df.copy()
    display["start_time"] = pd.to_datetime(
        display["start_time"], unit="ms", utc=True
    ).dt.strftime("%Y-%m-%d %H:%M")
    display = display.set_index("model")
    st.dataframe(display, width="stretch")


def _render_prediction_cards(
    best_runs: dict[str, RunSummary],
    models_with_source: dict[str, tuple[Any, str]],
    latest_row: pd.DataFrame,
) -> dict[str, int]:
    predictions: dict[str, int] = {}
    cols = st.columns(max(1, len(best_runs)))
    for col, model_name in zip(cols, best_runs.keys()):
        model, source = models_with_source[model_name]
        with col:
            if model is None:
                st.warning(f"{model_name}: model unavailable ({source})")
                continue
            pred = int(model.predict(latest_row)[0])
            try:
                proba = float(model.predict_proba(latest_row)[0, 1])
            except Exception:
                proba = None

            predictions[model_name] = pred
            direction = "UP" if pred == 1 else "DOWN"
            arrow = ":arrow_up:" if pred == 1 else ":arrow_down:"
            st.metric(
                label=f"{model_name.upper()} {arrow}",
                value=direction,
                delta=f"P(up) = {proba:.1%}" if proba is not None else None,
                delta_color="off",
            )
            run = best_runs[model_name]
            acc = run.metrics.get("accuracy")
            auc = run.metrics.get("roc_auc")
            caption_parts = []
            if acc is not None:
                caption_parts.append(f"acc {acc:.1%}")
            if auc is not None:
                caption_parts.append(f"auc {auc:.2f}")
            if caption_parts:
                st.caption("best run · " + " · ".join(caption_parts))
            if source != "mlflow":
                st.caption(f":warning: `{source}` (may differ from best run)")
    return predictions


def _render_disagreement_banner(predictions: dict[str, int]) -> None:
    uniq = set(predictions.values())
    if len(uniq) > 1:
        parts = [
            f"**{name}** = {'UP' if p == 1 else 'DOWN'}"
            for name, p in predictions.items()
        ]
        st.info(":warning: Models disagree — " + " · ".join(parts))
    elif uniq == {1}:
        st.success("All models agree: **UP**")
    elif uniq == {0}:
        st.warning("All models agree: **DOWN**")


def _metrics_bar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        values = [row.get(m) for m in COMPARISON_METRICS]
        fig.add_trace(
            go.Bar(
                name=str(row["model"]),
                x=list(COMPARISON_METRICS),
                y=values,
                text=[
                    f"{v:.2f}" if isinstance(v, (int, float)) and pd.notna(v) else ""
                    for v in values
                ],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1], title="score"),
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            x=0,
        ),
    )
    return fig


def _roc_overlay(diag_by_model: dict[str, ModelDiagnostics]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="chance",
            showlegend=True,
        )
    )
    for name, diag in diag_by_model.items():
        auc_label = (
            f" (AUC={diag.roc_auc:.2f})" if diag.roc_auc is not None else ""
        )
        fig.add_trace(
            go.Scatter(
                x=diag.fpr,
                y=diag.tpr,
                mode="lines",
                name=f"{name}{auc_label}",
                line=dict(width=2),
            )
        )
    fig.update_layout(
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            x=0,
        ),
    )
    return fig


def _render_confusion_matrices(
    diag_by_model: dict[str, ModelDiagnostics],
) -> None:
    cols = st.columns(max(1, len(diag_by_model)))
    for col, (name, diag) in zip(cols, diag_by_model.items()):
        with col:
            st.markdown(f"**{name}** · accuracy {diag.accuracy:.1%}")
            cm_df = pd.DataFrame(
                diag.confusion_matrix,
                index=["true_down", "true_up"],
                columns=["pred_down", "pred_up"],
            )
            st.table(cm_df)


def _feature_importance_chart(
    rf_model: Any,
) -> go.Figure | None:
    fi = feature_importance(rf_model, FEATURE_COLUMNS)
    if fi is None:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=fi["importance"][::-1],
            y=fi["feature"][::-1],
            orientation="h",
            marker=dict(color="steelblue"),
            text=[f"{v:.3f}" for v in fi["importance"][::-1]],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Importance (impurity-based)",
        height=max(320, 24 * len(fi) + 80),
        margin=dict(l=10, r=40, t=10, b=10),
    )
    return fig


def _render_all_runs_expander(
    runs_by_model: dict[str, list[RunSummary]],
) -> None:
    with st.expander("All runs (audit trail)", expanded=False):
        for name, runs in runs_by_model.items():
            st.markdown(f"**{name}** — {len(runs)} run(s)")
            table = runs_dataframe(runs)
            if "start_time" in table.columns and len(table) > 0:
                table["start_time"] = pd.to_datetime(
                    table["start_time"], unit="ms", utc=True
                ).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(table, width="stretch")


# ---------------------------------------------------------------------------
# Drift section
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _cached_drift_result() -> "DriftResult | None":
    features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        return None
    try:
        return check_drift_from_features(features_path)
    except (ValueError, Exception):
        return None


def _drift_pvalue_chart(result: "DriftResult") -> go.Figure:
    colors = [
        "crimson" if d else "seagreen" for d in result.is_drift_per_feature
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=result.p_values,
            y=result.feature_names,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:.3f}" for p in result.p_values],
            textposition="outside",
        )
    )
    fig.add_vline(
        x=result.p_val_threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"threshold={result.p_val_threshold}",
    )
    fig.update_layout(
        xaxis_title="p-value (KS test)",
        height=max(300, 26 * len(result.feature_names) + 80),
        margin=dict(l=10, r=60, t=10, b=10),
    )
    return fig


def _render_drift_section() -> None:
    st.header("Data Drift (Alibi Detect)")
    result = _cached_drift_result()
    if result is None:
        st.info("Drift detection unavailable — run `dvc repro` first.")
        return

    if result.is_drift:
        st.error("Drift detected — market regime may have shifted.")
    else:
        st.success("No drift detected — feature distributions are stable.")

    st.caption(
        f"Reference: {result.n_reference} rows | "
        f"Test window: {result.n_test} rows | "
        f"KS threshold: {result.p_val_threshold}"
    )
    st.plotly_chart(_drift_pvalue_chart(result), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("CryptoProfitMaxxing")
    st.caption(
        "MLOps pipeline demo — CoinGecko + DVC + MLflow + scikit-learn + Streamlit"
    )

    df = _cached_features()
    runs_by_model = _cached_runs_by_model(DEFAULT_EXPERIMENT_NAME)

    _render_sidebar(runs_by_model, df)

    if df.empty:
        st.warning(
            "No features CSV found. Run `dvc repro` to build the pipeline, "
            "then `python train.py --model logreg` and `python train.py --model rf`."
        )
        st.stop()

    _render_header(df)
    st.subheader("BTC / USD — close price")
    st.plotly_chart(_price_chart(df), width="stretch")

    st.header("Model comparison")

    if not runs_by_model:
        st.warning(
            f"No runs found in MLflow experiment `{DEFAULT_EXPERIMENT_NAME}`. "
            "Run `python train.py --model logreg` then `python train.py --model rf`."
        )
        st.stop()

    best_runs = best_run_per_model(runs_by_model)

    st.subheader("Best run per model")
    metrics_df = metrics_dataframe(best_runs)
    _render_metrics_table(metrics_df)

    models_with_source: dict[str, tuple[Any, str]] = {}
    for name, run in best_runs.items():
        models_with_source[name] = _cached_model_for_run(run.run_id, name)

    st.subheader("Next-day prediction")
    latest_row = df[FEATURE_COLUMNS].iloc[[-1]]
    predictions = _render_prediction_cards(
        best_runs, models_with_source, latest_row
    )
    if len(predictions) >= 2:
        _render_disagreement_banner(predictions)
    st.caption(
        f"Prediction horizon: day after {pd.to_datetime(df['timestamp'].iloc[-1]).date()}"
    )

    if len(best_runs) < 2:
        st.info(
            "Only one model available — train another with "
            "`python train.py --model <name>` for full comparison."
        )
        _render_all_runs_expander(runs_by_model)
        return

    st.subheader("Metric comparison")
    st.plotly_chart(_metrics_bar_chart(metrics_df), width="stretch")

    tune_info = tuning_summary(runs_by_model)
    has_tuned = any(v["total_tuned_runs"] > 0 for v in tune_info.values())
    if has_tuned:
        st.subheader("Hyperparameter tuning (Ray Tune)")
        tune_cols = st.columns(len(tune_info))
        for col, (model, info) in zip(tune_cols, tune_info.items()):
            with col:
                tuned_n = info["total_tuned_runs"]
                manual_n = info["total_manual_runs"]
                st.metric(f"{model} trials", tuned_n)
                st.caption(f"{manual_n} manual run(s)")

    st.subheader("Diagnostics on held-out test split")
    X_all = df[FEATURE_COLUMNS].copy()
    y_all = df[TARGET_COLUMN].astype(int)
    split = chronological_split(X_all, y_all, test_frac=0.2)

    diag_by_model: dict[str, ModelDiagnostics] = {}
    for name, (model, _) in models_with_source.items():
        if model is None:
            continue
        diag_by_model[name] = evaluate_on_split(
            model, split.X_test, split.y_test
        )

    if diag_by_model:
        st.markdown("**ROC curves**")
        st.plotly_chart(
            _roc_overlay(diag_by_model), width="stretch"
        )
        st.markdown("**Confusion matrices**")
        _render_confusion_matrices(diag_by_model)

    rf_entry = models_with_source.get("rf")
    if rf_entry is not None and rf_entry[0] is not None:
        fig = _feature_importance_chart(rf_entry[0])
        if fig is not None:
            st.subheader("Random Forest feature importance")
            st.plotly_chart(fig, width="stretch")

    if _DRIFT_AVAILABLE:
        _render_drift_section()

    _render_all_runs_expander(runs_by_model)

    with st.expander("About this dashboard"):
        st.markdown(
            """
            **CryptoProfitMaxxing** — MLOps pipeline demo.

            Pipeline:
            1. `dvc repro` — ingest CoinGecko BTC data and compute technical
               indicators (RSI, MACD, SMA, EMA).
            2. `python train.py --model {logreg,rf,lstm}` — train a classifier on
               a chronological 80/20 split. Params, metrics, and the sklearn
               artifact are logged to the local MLflow store; the run is
               registered to `crypto_trend_baseline`.
            3. `python tune.py --model {logreg,rf}` — Ray Tune hyperparameter
               search. Each trial is logged to MLflow and tagged for
               identification.
            4. This dashboard queries MLflow for the best run per model
               (ROC AUC → accuracy → start time), loads both, and compares
               them side by side. Click **Refresh from MLflow** in the
               sidebar after a new training run to invalidate caches.

            5. `python scripts/check_drift.py` — Alibi Detect KS-test
               drift monitoring on feature distributions.

            Tools: CoinGecko API, DVC, MLflow, Ray Tune, GitHub Actions,
            Streamlit, Alibi Detect.
            """
        )


if __name__ == "__main__":
    main()
