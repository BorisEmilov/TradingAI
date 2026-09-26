"""Swing points and BOS/CHoCH (break/change of structure) detection.

Both are causal: a swing pivot is only emitted once `right` bars after it have
CLOSED (you cannot know bar i is a local high until you've seen the full
high/low of what comes after it -- a still-forming bar's high/low isn't
final yet), and a structure break is only emitted at the CLOSE of the bar
whose close price actually crosses a swing level that was already confirmed
by then. Every event timestamp below is therefore a bar's CLOSE time
(`ts + TF_DURATION[timeframe]`), never its open time -- using the open time
would make the event look "confirmed" a full bar earlier than the data that
confirms it actually existed.
"""

from __future__ import annotations

from trader.events import TF_DURATION, MarketEvent


def detect_swings(df, timeframe: str, left: int, right: int) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    ts = df["timestamp"]
    close_ts = ts + TF_DURATION[timeframe]
    n = len(df)

    for i in range(left, n - right):
        if (highs[i] > highs[i - left : i]).all() and (highs[i] > highs[i + 1 : i + right + 1]).all():
            events.append(
                MarketEvent.point(
                    kind="swing_high",
                    timeframe=timeframe,
                    timestamp=close_ts.iloc[i + right],
                    direction="neutral",
                    price=float(highs[i]),
                    pivot_timestamp=ts.iloc[i],
                )
            )
        if (lows[i] < lows[i - left : i]).all() and (lows[i] < lows[i + 1 : i + right + 1]).all():
            events.append(
                MarketEvent.point(
                    kind="swing_low",
                    timeframe=timeframe,
                    timestamp=close_ts.iloc[i + right],
                    direction="neutral",
                    price=float(lows[i]),
                    pivot_timestamp=ts.iloc[i],
                )
            )

    events.sort(key=lambda e: e.timestamp)
    return events


def detect_structure_breaks(df, timeframe: str, swings: list[MarketEvent]) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    ts = df["timestamp"]
    close_ts = ts + TF_DURATION[timeframe]
    closes = df["close"].to_numpy()

    swings_sorted = sorted(swings, key=lambda e: e.timestamp)
    swing_idx = 0

    trend: str | None = None
    last_swing_high: float | None = None
    last_swing_low: float | None = None
    broken_high = False
    broken_low = False

    for i in range(len(df)):
        current_ts = close_ts.iloc[i]

        while swing_idx < len(swings_sorted) and swings_sorted[swing_idx].timestamp <= current_ts:
            sw = swings_sorted[swing_idx]
            if sw.kind == "swing_high":
                last_swing_high = sw.price
                broken_high = False
            else:
                last_swing_low = sw.price
                broken_low = False
            swing_idx += 1

        close = closes[i]

        if last_swing_high is not None and not broken_high and close > last_swing_high:
            kind = "choch" if trend == "down" else "bos"
            events.append(
                MarketEvent.point(
                    kind=kind,
                    timeframe=timeframe,
                    timestamp=current_ts,
                    direction="bullish",
                    price=float(close),
                    broken_level=last_swing_high,
                )
            )
            trend = "up"
            broken_high = True

        if last_swing_low is not None and not broken_low and close < last_swing_low:
            kind = "choch" if trend == "up" else "bos"
            events.append(
                MarketEvent.point(
                    kind=kind,
                    timeframe=timeframe,
                    timestamp=current_ts,
                    direction="bearish",
                    price=float(close),
                    broken_level=last_swing_low,
                )
            )
            trend = "down"
            broken_low = True

    return events
