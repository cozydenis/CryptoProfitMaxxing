"""Leak-free target labeling for next-horizon price direction.

The target for row ``t`` is based on ``close[t+horizon]`` vs ``close[t]``. The
last ``horizon`` rows are dropped because their target cannot be computed.
Feature columns at row ``t`` must only use data up to and including ``t`` — the
indicators in ``indicators.py`` are backward-looking so this holds by
construction.
"""

from __future__ import annotations

import pandas as pd


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return a new DataFrame with a ``target`` column appended.

    ``target[t] = 1`` if ``close[t+horizon] > close[t]``, else ``0``. The last
    ``horizon`` rows are dropped because their target is undefined.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if "close" not in df.columns:
        raise ValueError("add_target requires a 'close' column")
    if len(df) <= horizon:
        raise ValueError(
            f"DataFrame length ({len(df)}) must exceed horizon ({horizon})"
        )

    out = df.copy()
    future_close = out["close"].shift(-horizon)
    out["target"] = (future_close > out["close"]).astype("int64")
    out = out.iloc[:-horizon].reset_index(drop=True)
    return out
