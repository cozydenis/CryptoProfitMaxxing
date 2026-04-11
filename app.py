"""Streamlit dashboard for the CryptoProfitMaxxing baseline pipeline.

Week 9 skeleton: loads the most recent features CSV, picks a model from the
MLflow registry (falls back to the local joblib artifact), renders a BTC price
chart, and shows a next-day UP / DOWN prediction with class probability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DEFAULT_REGISTERED_MODEL_NAME,
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

st.set_page_config(
    page_title="CryptoProfitMaxxing",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[object, str]:
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{DEFAULT_REGISTERED_MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        return model, f"MLflow registry ({model_uri})"
    except Exception:
        pass

    for candidate in ("baseline_rf.pkl", "baseline_logreg.pkl"):
        path = MODELS_DIR / candidate
        if path.exists():
            return joblib.load(path), f"joblib fallback ({path.name})"

    return None, "no model found"


def _load_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


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
    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["sma_20"],
                mode="lines",
                name="SMA 20",
                line=dict(width=1, dash="dash"),
            )
        )
    if "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["sma_50"],
                mode="lines",
                name="SMA 50",
                line=dict(width=1, dash="dot"),
            )
        )
    fig.update_layout(
        title="BTC / USD — close price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def main() -> None:
    st.title("CryptoProfitMaxxing")
    st.caption(
        "MLOps pipeline demo — CoinGecko + DVC + MLflow + scikit-learn + Streamlit"
    )

    df = load_features()
    if df.empty:
        st.warning(
            "No features CSV found. Run `dvc repro` then `python train.py --model logreg`."
        )
        st.stop()

    model, source = load_model()

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest close (USD)", f"${df['close'].iloc[-1]:,.2f}")
    col1.caption(f"as of {df['timestamp'].iloc[-1].date()}")
    col2.metric(
        "Rows in features CSV",
        f"{len(df):,}",
        help="After dropping NaN rows from rolling windows and the horizon tail",
    )
    col3.metric("Model source", source.split(" ")[0])
    col3.caption(source)

    st.plotly_chart(_price_chart(df), width="stretch")

    st.subheader("Next-day prediction")
    if model is None:
        st.info("No trained model found yet. Run `python train.py --model logreg`.")
    else:
        latest_features = df[FEATURE_COLUMNS].iloc[[-1]]
        pred = int(model.predict(latest_features)[0])
        try:
            proba = float(model.predict_proba(latest_features)[0, 1])
        except Exception:
            proba = None
        label = "UP" if pred == 1 else "DOWN"
        emoji = ":arrow_up:" if pred == 1 else ":arrow_down:"
        cols = st.columns(2)
        cols[0].metric(f"Direction {emoji}", label)
        if proba is not None:
            cols[1].metric("P(up)", f"{proba:.1%}")

    st.subheader("Last training run metrics")
    metrics = _load_metrics()
    if not metrics:
        st.info("No `models/metrics.json` yet — run `python train.py` first.")
    else:
        display = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        st.json(display)
        if "confusion_matrix" in metrics:
            st.caption("Confusion matrix (rows=true, cols=pred)")
            st.table(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["true_down", "true_up"],
                    columns=["pred_down", "pred_up"],
                )
            )

    with st.expander("About this dashboard"):
        st.markdown(
            """
            This is a **Week 9 skeleton** for the CryptoProfitMaxxing MLOps demo.

            The pipeline:

            1. `dvc repro` — ingest CoinGecko BTC data and compute technical
               indicators (RSI, MACD, SMA, EMA).
            2. `python train.py --model logreg` / `--model rf` — train a
               classifier on a chronological 80/20 split, log params + metrics +
               artifacts to MLflow, register the model in the MLflow registry.
            3. This dashboard loads the latest registered model and predicts
               whether BTC will be higher tomorrow than today.

            Planned for Week 10+: side-by-side model comparison from the MLflow
            registry, ETH support, hyperparameter tuning with Ray Tune, drift
            detection with Alibi Detect.
            """
        )


if __name__ == "__main__":
    main()
