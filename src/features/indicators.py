"""Technical indicators computed from daily close + volume.

All functions return new DataFrames — inputs are never mutated. We compute
indicators with hand-rolled formulas rather than `pandas-ta` to avoid a
dependency that is in light maintenance and has flaky installs on newer Python
versions. The formulas here match the standard TA definitions used by
`pandas-ta` / `ta-lib`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with technical indicator columns appended.

    Input must have ``close`` and ``volume`` columns (and typically
    ``timestamp``). The output preserves input columns and adds:
    ``return_1d, sma_10, sma_20, sma_50, ema_12, ema_26, rsi_14, macd,
    macd_signal, macd_hist, volume_change_1d``. Rows with NaNs from rolling
    windows are NOT dropped here — that is the caller's responsibility.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("add_indicators requires 'close' and 'volume' columns")

    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    out["return_1d"] = close.pct_change()
    out["sma_10"] = close.rolling(window=10).mean()
    out["sma_20"] = close.rolling(window=20).mean()
    out["sma_50"] = close.rolling(window=50).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["rsi_14"] = _rsi(close, window=14)

    macd_df = _macd(close)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["macd_signal"]
    out["macd_hist"] = macd_df["macd_hist"]

    out["volume_change_1d"] = volume.pct_change().replace(
        [np.inf, -np.inf], np.nan
    )

    return out
