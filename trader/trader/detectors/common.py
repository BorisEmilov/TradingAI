"""Shared zone lifecycle: mitigation (first touch), confirmation, and
invalidation (a close all the way through the zone).

Used by both order_blocks.py and fvg.py since OB/IOB and FVG/IFVG share the
same lifecycle shape. Per prompt-fix-invalidacion-obfvg.md: a touch that gets
respected (price rebounds without closing through) is not disqualifying --
that's the standard ICT reading of a "respected" zone, not a reason to
discard it. Only a genuine close-through invalidates a zone, and even then
only counts if it doesn't happen too fast to have been tradeable (a break
within `grace_period` of the first touch means the zone was never actually
usable, so it's excluded from formation onward, not just from the break).
"""

from __future__ import annotations

import pandas as pd

from trader.events import TF_DURATION, MarketEvent

INVERTED_KIND = {
    "order_block_bullish": "inverted_order_block_bullish",
    "order_block_bearish": "inverted_order_block_bearish",
    "fair_value_gap_bullish": "inverted_fair_value_gap_bullish",
    "fair_value_gap_bearish": "inverted_fair_value_gap_bearish",
}


def apply_mitigation(
    zones: list[MarketEvent], df, timeframe: str, grace_period: pd.Timedelta
) -> tuple[list[MarketEvent], list[MarketEvent]]:
    """Scans every candle strictly after a zone's formation (not just the
    first touch -- scanning stopped there in an earlier version, which meant
    a break several candles after the first touch was never detected at
    all) for the first touch and the first genuine close-through. Returns
    (zones_with_lifecycle_fields, inversion_events).

    A zone's own `.timestamp` is its confirming bar's CLOSE time (see the
    detectors that produce it), so the lookup index below is built from
    close times too -- indexing by open time here would never find the zone
    at all in a market with any gaps (weekends, holidays).
    """
    close_ts = df["timestamp"] + TF_DURATION[timeframe]
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)
    ts_index = {t: i for i, t in enumerate(close_ts)}

    out: list[MarketEvent] = []
    inversions: list[MarketEvent] = []

    for z in zones:
        start_i = ts_index.get(z.timestamp)
        touched_at: pd.Timestamp | None = None
        broken_at: pd.Timestamp | None = None

        if start_i is not None:
            for k in range(start_i + 1, n):
                if touched_at is None and lows[k] <= z.price_high and highs[k] >= z.price_low:
                    touched_at = close_ts.iloc[k]

                is_break = (z.direction == "bullish" and closes[k] < z.price_low) or (
                    z.direction == "bearish" and closes[k] > z.price_high
                )
                if is_break:
                    broken_at = close_ts.iloc[k]
                    if touched_at is None:
                        touched_at = broken_at  # broke without a prior recorded touch (rare gap case)
                    inv_kind = INVERTED_KIND.get(z.kind)
                    if inv_kind:
                        inversions.append(
                            MarketEvent.zone(
                                kind=inv_kind,
                                timeframe=timeframe,
                                timestamp=broken_at,
                                direction="bearish" if z.direction == "bullish" else "bullish",
                                price_high=z.price_high,
                                price_low=z.price_low,
                                origin_timestamp=z.timestamp,
                            )
                        )
                    break  # first genuine break ends the zone's life; nothing after it matters

        if touched_at is None:
            confirmed_at = z.timestamp  # never touched -- eligible from formation
        elif broken_at is None:
            confirmed_at = touched_at + grace_period  # touched, never broke (in the observed window)
        elif (broken_at - touched_at) >= grace_period:
            confirmed_at = touched_at + grace_period  # held long enough before eventually breaking
        else:
            confirmed_at = None  # broke too fast after the touch -- never a valid POI

        out.append(
            MarketEvent(
                kind=z.kind,
                timeframe=z.timeframe,
                timestamp=z.timestamp,
                direction=z.direction,
                price=z.price,
                price_high=z.price_high,
                price_low=z.price_low,
                mitigated=touched_at is not None,
                mitigated_at=touched_at,
                confirmed_at=confirmed_at,
                broken_at=broken_at,
                meta=z.meta,
            )
        )

    return out, inversions
