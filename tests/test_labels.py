"""Tests for leak-free target labeling — the single highest-risk module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.labels import add_target


def test_target_known_values_horizon_1():
    df = pd.DataFrame({"close": [100.0, 110.0, 105.0, 120.0]})
    out = add_target(df, horizon=1)
    # last row dropped because its target is undefined
    assert len(out) == 3
    # target[0] = 1 (110 > 100), target[1] = 0 (105 < 110), target[2] = 1 (120 > 105)
    assert list(out["target"]) == [1, 0, 1]


def test_target_horizon_2():
    df = pd.DataFrame({"close": [100.0, 50.0, 110.0, 80.0, 200.0]})
    out = add_target(df, horizon=2)
    # last 2 rows dropped
    assert len(out) == 3
    # target[0] = 1 (110 > 100), target[1] = 1 (80 > 50), target[2] = 1 (200 > 110)
    assert list(out["target"]) == [1, 1, 1]


def test_target_no_future_leakage():
    """Critical: features at row t must not change based on row t+1's close."""
    df = pd.DataFrame({"close": [100.0, 110.0, 105.0, 120.0], "feature": [1, 2, 3, 4]})
    mutated = df.copy()
    mutated.loc[3, "close"] = 999.0  # change only future close

    out_original = add_target(df, horizon=1)
    out_mutated = add_target(mutated, horizon=1)

    # Changing the last row's close should ONLY affect row 2's target
    # (row 2 is now the last, and row 2's target depends on row 3's close).
    # Rows 0 and 1 should be identical.
    pd.testing.assert_series_equal(
        out_original.iloc[:2]["target"], out_mutated.iloc[:2]["target"]
    )


def test_add_target_does_not_mutate_input():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    before = df.copy()
    _ = add_target(df, horizon=1)
    pd.testing.assert_frame_equal(df, before)


def test_target_dtype_is_int():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    out = add_target(df, horizon=1)
    assert out["target"].dtype == "int64"


def test_raises_on_invalid_horizon():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="horizon must be"):
        add_target(df, horizon=0)


def test_raises_on_missing_close():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    with pytest.raises(ValueError, match="close"):
        add_target(df, horizon=1)


def test_raises_when_length_insufficient():
    df = pd.DataFrame({"close": [1.0]})
    with pytest.raises(ValueError, match="length"):
        add_target(df, horizon=1)
