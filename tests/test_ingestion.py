"""Tests for CoinGecko ingestion (mocked — no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.ingestion.coingecko import (
    _build_frame,
    fetch_market_chart,
    validate_frame,
)


def test_build_frame_shape(synthetic_market_chart):
    df = _build_frame(synthetic_market_chart)
    assert list(df.columns) == ["timestamp", "close", "volume"]
    assert len(df) == len(synthetic_market_chart["prices"])
    assert df["close"].notna().all()
    assert df["volume"].notna().all()


def test_build_frame_sorted_monotonic(synthetic_market_chart):
    df = _build_frame(synthetic_market_chart)
    assert df["timestamp"].is_monotonic_increasing


def test_build_frame_raises_on_empty():
    with pytest.raises(ValueError, match="no prices"):
        _build_frame({"prices": [], "total_volumes": []})


def test_build_frame_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        _build_frame(
            {
                "prices": [[1_700_000_000_000, 100.0]],
                "total_volumes": [],
            }
        )


def test_fetch_market_chart_uses_injected_client(synthetic_market_chart):
    client = MagicMock()
    client.get_coin_market_chart_by_id.return_value = synthetic_market_chart
    df = fetch_market_chart(coin_id="bitcoin", days=30, client=client)
    client.get_coin_market_chart_by_id.assert_called_once_with(
        id="bitcoin", vs_currency="usd", days=30
    )
    assert len(df) == 30


def test_validate_frame_happy(synthetic_market_chart):
    df = _build_frame(synthetic_market_chart)
    validate_frame(df)


def test_validate_frame_missing_column():
    df = pd.DataFrame({"timestamp": [1, 2], "close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_frame(df)


def test_validate_frame_non_positive_close(synthetic_market_chart):
    df = _build_frame(synthetic_market_chart)
    df.loc[0, "close"] = 0.0
    with pytest.raises(ValueError, match="Non-positive"):
        validate_frame(df)


def test_validate_frame_duplicate_timestamps(synthetic_market_chart):
    df = _build_frame(synthetic_market_chart)
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    df = df.sort_values("timestamp").reset_index(drop=True)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_frame(df)
