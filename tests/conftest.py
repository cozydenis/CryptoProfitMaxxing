"""Shared pytest fixtures — synthetic OHLCV frames, no network calls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_market_chart() -> dict:
    """Return a CoinGecko-shaped market_chart response for 30 days."""
    rng = np.random.default_rng(seed=42)
    base_ts_ms = 1_700_000_000_000
    day_ms = 86_400_000
    prices = []
    volumes = []
    price = 50_000.0
    for i in range(30):
        price *= 1.0 + float(rng.normal(0.0, 0.02))
        prices.append([base_ts_ms + i * day_ms, price])
        volumes.append([base_ts_ms + i * day_ms, 1_000_000_000.0 + float(rng.normal(0, 1e8))])
    return {"prices": prices, "total_volumes": volumes, "market_caps": []}


@pytest.fixture
def synthetic_price_df() -> pd.DataFrame:
    """Return a 200-row daily close+volume DataFrame with a modest upward drift."""
    rng = np.random.default_rng(seed=7)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    returns = rng.normal(loc=0.001, scale=0.02, size=n)
    close = 30_000.0 * np.cumprod(1.0 + returns)
    volume = rng.uniform(8e8, 1.2e9, size=n)
    return pd.DataFrame(
        {"timestamp": timestamps, "close": close, "volume": volume}
    )
