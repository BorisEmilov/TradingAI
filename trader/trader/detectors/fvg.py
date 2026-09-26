"""Fair Value Gap (FVG): 3-candle imbalance, C1.high < C3.low (bullish) or
C3.high < C1.low (bearish). Confirmed at C3's close -- NOT before, since the
gap isn't real until the 3rd candle has actually printed (an earlier version
of this project's sibling looked ahead by trusting the gap as soon as C2
formed).
"""

from __future__ import annotations

from trader.events import TF_DURATION, MarketEvent


def detect_fvg(df, timeframe: str, min_gap_pct: float = 0.0) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    close_ts = df["timestamp"] + TF_DURATION[timeframe]
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    n = len(df)

    for i in range(2, n):
        c1_high, c1_low = h[i - 2], l[i - 2]
        c3_high, c3_low = h[i], l[i]

        if c1_high < c3_low:
            gap_pct = (c3_low - c1_high) / c3_low * 100.0
            if gap_pct >= min_gap_pct:
                events.append(
                    MarketEvent.zone(
                        kind="fair_value_gap_bullish",
                        timeframe=timeframe,
                        timestamp=close_ts.iloc[i],
                        direction="bullish",
                        price_high=float(c3_low),
                        price_low=float(c1_high),
                    )
                )

        if c3_high < c1_low:
            gap_pct = (c1_low - c3_high) / c1_low * 100.0
            if gap_pct >= min_gap_pct:
                events.append(
                    MarketEvent.zone(
                        kind="fair_value_gap_bearish",
                        timeframe=timeframe,
                        timestamp=close_ts.iloc[i],
                        direction="bearish",
                        price_high=float(c1_low),
                        price_low=float(c3_high),
                    )
                )

    return events
