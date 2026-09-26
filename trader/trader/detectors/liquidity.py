"""Liquidity concepts: sweeps of swing highs/lows, equal highs/lows, turtle
soup (false breakout of a rolling N-bar range), and sharp turns (back-to-back
opposite displacement candles signaling exhaustion).

Sweeps and turtle soup are confirmed on the SAME candle that pierces-and-closes
back inside the level -- causal because the wick and the close both belong to
the one candle being evaluated, and the level being swept was already known
(confirmed) as of an earlier candle.
"""

from __future__ import annotations

import pandas as pd

from trader.events import TF_DURATION, MarketEvent


def _latest_known(levels: list[MarketEvent], as_of: pd.Timestamp) -> MarketEvent | None:
    best = None
    for lvl in levels:
        if lvl.timestamp <= as_of:
            best = lvl
        else:
            break
    return best


def detect_liquidity_sweeps(
    df, timeframe: str, swings: list[MarketEvent], min_wick_pct: float
) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    close_ts = df["timestamp"] + TF_DURATION[timeframe]
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)

    swing_highs = sorted((s for s in swings if s.kind == "swing_high"), key=lambda e: e.timestamp)
    swing_lows = sorted((s for s in swings if s.kind == "swing_low"), key=lambda e: e.timestamp)

    for i in range(n):
        current_ts = close_ts.iloc[i]
        candle_range = h[i] - l[i]
        if candle_range <= 0:
            continue

        sh = _latest_known(swing_highs, current_ts)
        if sh is not None and h[i] > sh.price and c[i] < sh.price:
            wick = h[i] - max(o[i], c[i])
            if wick / candle_range * 100.0 >= min_wick_pct:
                events.append(
                    MarketEvent.point(
                        kind="liquidity_sweep_bearish",
                        timeframe=timeframe,
                        timestamp=current_ts,
                        direction="bearish",
                        price=float(sh.price),
                        swept_level_timestamp=sh.timestamp,
                    )
                )

        sl = _latest_known(swing_lows, current_ts)
        if sl is not None and l[i] < sl.price and c[i] > sl.price:
            wick = min(o[i], c[i]) - l[i]
            if wick / candle_range * 100.0 >= min_wick_pct:
                events.append(
                    MarketEvent.point(
                        kind="liquidity_sweep_bullish",
                        timeframe=timeframe,
                        timestamp=current_ts,
                        direction="bullish",
                        price=float(sl.price),
                        swept_level_timestamp=sl.timestamp,
                    )
                )

    return events


def _cluster(sorted_swings: list[MarketEvent], kind: str, timeframe: str, tolerance_pct: float) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    i = 0
    n = len(sorted_swings)
    direction = "bearish" if kind == "equal_highs" else "bullish"

    while i < n:
        j = i
        anchor_price = sorted_swings[i].price
        group = [sorted_swings[i]]
        while j + 1 < n and abs(sorted_swings[j + 1].price - anchor_price) / anchor_price * 100.0 <= tolerance_pct:
            j += 1
            group.append(sorted_swings[j])
        if len(group) >= 2:
            confirm_ts = max(s.timestamp for s in group)
            avg_price = sum(s.price for s in group) / len(group)
            events.append(
                MarketEvent.point(
                    kind=kind,
                    timeframe=timeframe,
                    timestamp=confirm_ts,
                    direction=direction,
                    price=float(avg_price),
                    count=len(group),
                )
            )
        i = j + 1

    return events


def detect_equal_levels(swings: list[MarketEvent], timeframe: str, tolerance_pct: float) -> list[MarketEvent]:
    highs = sorted((s for s in swings if s.kind == "swing_high"), key=lambda e: e.price)
    lows = sorted((s for s in swings if s.kind == "swing_low"), key=lambda e: e.price)
    return _cluster(highs, "equal_highs", timeframe, tolerance_pct) + _cluster(lows, "equal_lows", timeframe, tolerance_pct)


def detect_turtle_soup(df, timeframe: str, lookback_bars: int, min_wick_pct: float) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    close_ts = df["timestamp"] + TF_DURATION[timeframe]
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)

    for i in range(lookback_bars, n):
        window_high = h[i - lookback_bars : i].max()
        window_low = l[i - lookback_bars : i].min()
        candle_range = h[i] - l[i]
        if candle_range <= 0:
            continue

        if h[i] > window_high and c[i] < window_high:
            wick = h[i] - max(o[i], c[i])
            if wick / candle_range * 100.0 >= min_wick_pct:
                events.append(
                    MarketEvent.point(
                        kind="turtle_soup_bearish",
                        timeframe=timeframe,
                        timestamp=close_ts.iloc[i],
                        direction="bearish",
                        price=float(window_high),
                        lookback_bars=lookback_bars,
                    )
                )

        if l[i] < window_low and c[i] > window_low:
            wick = min(o[i], c[i]) - l[i]
            if wick / candle_range * 100.0 >= min_wick_pct:
                events.append(
                    MarketEvent.point(
                        kind="turtle_soup_bullish",
                        timeframe=timeframe,
                        timestamp=close_ts.iloc[i],
                        direction="bullish",
                        price=float(window_low),
                        lookback_bars=lookback_bars,
                    )
                )

    return events


def detect_sharp_turns(df, timeframe: str, atr: pd.Series, displacement_atr_multiple: float) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    close_ts = df["timestamp"] + TF_DURATION[timeframe]
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    atr_arr = atr.to_numpy()
    n = len(df)

    for i in range(1, n):
        if pd.isna(atr_arr[i]) or pd.isna(atr_arr[i - 1]):
            continue
        prev_range = h[i - 1] - l[i - 1]
        cur_range = h[i] - l[i]
        prev_bullish = c[i - 1] > o[i - 1]
        cur_bullish = c[i] > o[i]
        strong_prev = prev_range >= displacement_atr_multiple * atr_arr[i - 1]
        strong_cur = cur_range >= displacement_atr_multiple * atr_arr[i]

        if strong_prev and strong_cur and prev_bullish != cur_bullish:
            direction = "bearish" if prev_bullish else "bullish"
            events.append(
                MarketEvent.point(
                    kind="sharp_turn",
                    timeframe=timeframe,
                    timestamp=close_ts.iloc[i],
                    direction=direction,
                    price=float(c[i]),
                )
            )

    return events
