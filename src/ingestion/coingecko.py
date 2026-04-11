"""CoinGecko ingestion — fetches daily close + volume for a single coin.

The CoinGecko free tier's /market_chart endpoint returns one datapoint per day
when `days >= 90`, structured as prices=[[ms, close], ...] and
total_volumes=[[ms, volume], ...]. OHLC endpoints return coarser granularity on
free tier (4-day candles for 365 days), so we use market_chart and work with
close+volume. This is sufficient for the TA indicators we rely on (RSI, MACD,
SMA, EMA on close; volume change on volume).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from pycoingecko import CoinGeckoAPI

from src.config import PRICE_COLUMNS


def _build_frame(market_chart: Dict[str, Any]) -> pd.DataFrame:
    """Convert a CoinGecko market_chart response into a tidy DataFrame.

    `market_chart` is expected to contain `prices` and `total_volumes`, each a
    list of [timestamp_ms, value] pairs aligned one-to-one.
    """
    prices = market_chart.get("prices") or []
    volumes = market_chart.get("total_volumes") or []
    if not prices:
        raise ValueError("CoinGecko response has no prices")
    if len(prices) != len(volumes):
        raise ValueError(
            f"prices ({len(prices)}) and total_volumes ({len(volumes)}) length mismatch"
        )

    price_df = pd.DataFrame(prices, columns=["ts_ms", "close"])
    vol_df = pd.DataFrame(volumes, columns=["ts_ms", "volume"])
    df = price_df.merge(vol_df, on="ts_ms", how="inner")
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.normalize()
    df = df[["timestamp", "close", "volume"]].copy()
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_market_chart(
    coin_id: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 365,
    client: CoinGeckoAPI | None = None,
) -> pd.DataFrame:
    """Fetch `days` of daily close+volume for `coin_id` from CoinGecko.

    Returns a DataFrame with columns ``timestamp, close, volume`` (see
    ``config.PRICE_COLUMNS``). The DataFrame is sorted ascending by timestamp,
    has no duplicate timestamps, and contains no nulls.
    """
    cg = client if client is not None else CoinGeckoAPI()
    response = cg.get_coin_market_chart_by_id(
        id=coin_id, vs_currency=vs_currency, days=days
    )
    return _build_frame(response)


def validate_frame(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` violates the expected ingestion schema."""
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if df[["close", "volume"]].isnull().any().any():
        raise ValueError("Null values in close/volume")
    if (df["close"] <= 0).any():
        raise ValueError("Non-positive close prices")
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonically increasing")
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamps")
