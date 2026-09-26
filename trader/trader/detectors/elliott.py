"""Best-effort Elliott Wave context: a causal percentage ZigZag over pivots,
then classic Fibonacci-ratio checks on the last 5 legs. This is explicitly
CONTEXT for bias, never a standalone entry signal (per the strategy spec) --
real-world wave counts are subjective even among professional analysts.

Rule-flags are summed as plain Python bools/ints on purpose: `numpy.bool_ +
numpy.bool_` performs logical OR (saturates at True), not arithmetic addition
-- a prior version of a sibling project silently scored a perfect 4/4-rule
impulse as 1/4 because of exactly this. Pivot prices here are cast to native
`float` at the ZigZag stage so every downstream comparison stays on plain
Python types.
"""

from __future__ import annotations

import pandas as pd

from trader.events import TF_DURATION, MarketEvent


def zigzag_pivots(df, timeframe: str, deviation_pct: float) -> list[tuple[pd.Timestamp, str, float]]:
    """Percentage ZigZag. The PRICE of each pivot comes from wherever the true
    extreme was, but the TIMESTAMP is the CONFIRMING bar's close (the bar
    whose data crossed the deviation threshold) -- that's the earliest
    instant this pivot could actually be known, which is often many bars
    after the extreme itself. Stamping it with the extreme bar's own
    timestamp instead (an earlier version of this function did) would be a
    direct look-ahead: it'd claim the pivot was knowable before the price
    action that actually confirms it had even happened.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close_ts = df["timestamp"] + TF_DURATION[timeframe]
    n = len(df)
    if n < 2:
        return []

    pivots: list[tuple[int, str, float]] = []
    up_trend: bool | None = None

    cand_high = float(high[0])
    cand_low = float(low[0])

    for i in range(1, n):
        if high[i] > cand_high:
            cand_high = float(high[i])
        if low[i] < cand_low:
            cand_low = float(low[i])

        if up_trend is None:
            drop_pct = (cand_high - low[i]) / cand_high * 100.0
            rise_pct = (high[i] - cand_low) / cand_low * 100.0
            if drop_pct >= deviation_pct and drop_pct >= rise_pct:
                pivots.append((i, "high", cand_high))
                up_trend = False
                cand_low = float(low[i])
            elif rise_pct >= deviation_pct:
                pivots.append((i, "low", cand_low))
                up_trend = True
                cand_high = float(high[i])
        elif up_trend:
            drop_pct = (cand_high - low[i]) / cand_high * 100.0
            if drop_pct >= deviation_pct:
                pivots.append((i, "high", cand_high))
                up_trend = False
                cand_low = float(low[i])
        else:
            rise_pct = (high[i] - cand_low) / cand_low * 100.0
            if rise_pct >= deviation_pct:
                pivots.append((i, "low", cand_low))
                up_trend = True
                cand_high = float(high[i])

    return [(close_ts.iloc[confirm_i], kind, price) for confirm_i, kind, price in pivots]


def elliott_wave_context(df, timeframe: str, deviation_pct: float) -> MarketEvent | None:
    pivots = zigzag_pivots(df, timeframe, deviation_pct)
    if len(pivots) < 6:
        return None

    p0, p1, p2, p3, p4, p5 = pivots[-6:]
    price0, price1, price2, price3, price4 = p0[2], p1[2], p2[2], p3[2], p4[2]
    ts5, _, price5 = p5

    bullish_impulse = price1 > price0 and price3 > price1 and price5 > price3
    bearish_impulse = price1 < price0 and price3 < price1 and price5 < price3
    if not (bullish_impulse or bearish_impulse):
        return None

    wave1 = abs(price1 - price0)
    wave2 = abs(price2 - price1)
    wave3 = abs(price3 - price2)
    wave4 = abs(price4 - price3)
    wave5 = abs(price5 - price4)

    rule_wave2_retrace = bool(0.382 <= (wave2 / wave1 if wave1 else 0.0) <= 0.886)
    rule_wave3_not_shortest = bool(wave3 >= wave1 and wave3 >= wave5)
    rule_wave4_no_overlap = bool(price4 > price1) if bullish_impulse else bool(price4 < price1)
    rule_wave4_retrace = bool(0.146 <= (wave4 / wave3 if wave3 else 0.0) <= 0.618)

    rules = [rule_wave2_retrace, rule_wave3_not_shortest, rule_wave4_no_overlap, rule_wave4_retrace]
    rules_passed = sum(1 for r in rules if r)
    confidence = rules_passed / float(len(rules))

    return MarketEvent.point(
        kind="elliott_impulse_context",
        timeframe=timeframe,
        timestamp=ts5,
        direction="bullish" if bullish_impulse else "bearish",
        price=float(price5),
        confidence=confidence,
        wave_position="5",
        rules_passed=rules_passed,
    )
