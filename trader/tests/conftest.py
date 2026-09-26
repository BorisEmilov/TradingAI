from __future__ import annotations

import pandas as pd


def build_candles(ohlc: list[tuple[float, float, float, float]], start: str = "2026-01-05 00:00:00", freq: str = "1h") -> pd.DataFrame:
    """ohlc: list of (open, high, low, close) tuples, oldest first."""
    ts = pd.date_range(start=pd.Timestamp(start, tz="UTC"), periods=len(ohlc), freq=freq)
    rows = [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in ohlc]
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", ts)
    return df
