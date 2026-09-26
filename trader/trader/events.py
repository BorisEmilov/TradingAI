"""Shared event/zone model and timeframe-alignment helpers.

Every detector emits `MarketEvent`s identified by `timestamp`, never by row
position. Row indices are never passed between DataFrames that were filtered
independently -- that pattern caused a real, costly look-ahead bug in an
earlier iteration of this project (a signal's row index computed against a
features DataFrame was used to slice a differently-shaped raw candles
DataFrame, silently offsetting every trade by ~200 bars). Timestamps are the
only cross-DataFrame identifier used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

TIMEFRAMES = ("D1", "H4", "H1", "M30", "M15", "M5")

TF_DURATION: dict[str, pd.Timedelta] = {
    "D1": pd.Timedelta(days=1),
    "H4": pd.Timedelta(hours=4),
    "H1": pd.Timedelta(hours=1),
    "M30": pd.Timedelta(minutes=30),
    "M15": pd.Timedelta(minutes=15),
    "M5": pd.Timedelta(minutes=5),
}


@dataclass(frozen=True)
class MarketEvent:
    """A detected structural event or zone, causal by construction.

    `timestamp` is when the event became KNOWABLE (e.g. for a 3-candle FVG,
    the close of the 3rd candle -- not the candle where the gap geometrically
    starts). Zones use `price_high`/`price_low`; point events set both equal
    to `price`.
    """

    kind: str
    timeframe: str
    timestamp: pd.Timestamp
    direction: str  # "bullish" | "bearish" | "neutral"
    price: float
    price_high: float
    price_low: float
    mitigated: bool = False
    mitigated_at: pd.Timestamp | None = None
    # Zone eligibility as a POI, per prompt-fix-invalidacion-obfvg.md: a touch
    # alone no longer disqualifies a zone -- only a genuine break (close
    # through the far boundary) does, and only once the touch-to-break gap is
    # too short to have been tradeable. `confirmed_at` is when the zone
    # becomes/became eligible (None = never confirmed -- broke too fast after
    # a touch to ever count); `broken_at` is when it's permanently invalidated
    # (None = never breaks in the observed window). Both are static facts
    # about the underlying candles, independent of any `as_of` -- causality is
    # enforced by the caller comparing them against `as_of` directly, not by
    # these fields changing shape.
    confirmed_at: pd.Timestamp | None = None
    broken_at: pd.Timestamp | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def point(
        kind: str,
        timeframe: str,
        timestamp: pd.Timestamp,
        direction: str,
        price: float,
        **meta: Any,
    ) -> "MarketEvent":
        return MarketEvent(
            kind=kind,
            timeframe=timeframe,
            timestamp=timestamp,
            direction=direction,
            price=price,
            price_high=price,
            price_low=price,
            meta=meta,
        )

    @staticmethod
    def zone(
        kind: str,
        timeframe: str,
        timestamp: pd.Timestamp,
        direction: str,
        price_high: float,
        price_low: float,
        **meta: Any,
    ) -> "MarketEvent":
        if price_low > price_high:
            price_high, price_low = price_low, price_high
        return MarketEvent(
            kind=kind,
            timeframe=timeframe,
            timestamp=timestamp,
            direction=direction,
            price=(price_high + price_low) / 2.0,
            price_high=price_high,
            price_low=price_low,
            meta=meta,
        )

    def overlaps(self, low: float, high: float) -> bool:
        return self.price_low <= high and self.price_high >= low


def ensure_utc_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the candle DataFrame contract shared by every module here."""
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candles DataFrame missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("timestamp column must be datetime64")
    if df["timestamp"].dt.tz is None:
        raise ValueError("timestamp column must be timezone-aware (UTC)")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("candles must be sorted ascending by timestamp")
    if df["timestamp"].duplicated().any():
        raise ValueError("candles must not contain duplicate timestamps")
    return df


def closed_candles_as_of(df: pd.DataFrame, timeframe: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return only the candles of `df` that are FULLY CLOSED by `as_of`.

    A candle with open time `t` on timeframe `tf` closes at `t + duration`.
    Excluding any candle still forming at `as_of` is what makes every
    downstream detector causal -- callers never see a partially-formed bar.
    """
    ensure_utc_sorted(df)
    duration = TF_DURATION[timeframe]
    close_time = df["timestamp"] + duration
    return df.loc[close_time <= as_of].reset_index(drop=True)


def events_up_to(events: list[MarketEvent], as_of: pd.Timestamp) -> list[MarketEvent]:
    """Filter events to only those knowable by `as_of` (their own timestamp)."""
    return [e for e in events if e.timestamp <= as_of]
