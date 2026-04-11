"""Tests for technical indicator computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import _macd, _rsi, add_indicators


def test_add_indicators_does_not_mutate_input(synthetic_price_df):
    before = synthetic_price_df.copy()
    _ = add_indicators(synthetic_price_df)
    pd.testing.assert_frame_equal(synthetic_price_df, before)


def test_add_indicators_adds_expected_columns(synthetic_price_df):
    out = add_indicators(synthetic_price_df)
    expected = {
        "return_1d",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "volume_change_1d",
    }
    assert expected.issubset(out.columns)


def test_sma_known_values():
    df = pd.DataFrame(
        {
            "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "volume": [100.0] * 11,
        }
    )
    out = add_indicators(df)
    # sma_10 at index 9 should be mean(1..10) = 5.5
    assert out["sma_10"].iloc[9] == pytest.approx(5.5)
    # sma_10 at index 10 should be mean(2..11) = 6.5
    assert out["sma_10"].iloc[10] == pytest.approx(6.5)
    # rows before window are NaN
    assert out["sma_10"].iloc[:9].isna().all()


def test_return_1d_known_values():
    df = pd.DataFrame(
        {"close": [100.0, 110.0, 99.0], "volume": [1.0, 1.0, 1.0]}
    )
    out = add_indicators(df)
    assert np.isnan(out["return_1d"].iloc[0])
    assert out["return_1d"].iloc[1] == pytest.approx(0.10)
    assert out["return_1d"].iloc[2] == pytest.approx(-0.10)


def test_rsi_bounds(synthetic_price_df):
    out = add_indicators(synthetic_price_df)
    rsi = out["rsi_14"].dropna()
    assert ((rsi >= 0.0) & (rsi <= 100.0)).all()


def test_rsi_monotonic_rise_saturates():
    # strictly monotonic rising close → RSI should head to 100
    close = pd.Series(np.arange(1, 60, dtype=float))
    rsi = _rsi(close, window=14).dropna()
    assert rsi.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_macd_zero_when_constant():
    close = pd.Series([100.0] * 60)
    macd_df = _macd(close)
    assert macd_df["macd"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert macd_df["macd_signal"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert macd_df["macd_hist"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_add_indicators_raises_on_missing_columns():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="close.*volume"):
        add_indicators(df)
