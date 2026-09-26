"""Order Blocks (OB): last opposite-colored candle before a displacement move
that produced a confirmed structure break (BOS/CHoCH). Requires the break
candle's range to be a multiple of ATR (a real impulse, not noise) -- an OB
formed by a weak, low-conviction break is not a valid OB.
"""

from __future__ import annotations

import pandas as pd

from trader.events import TF_DURATION, MarketEvent

_MAX_LOOKBACK = 10


def detect_order_blocks(
    df,
    timeframe: str,
    structure_events: list[MarketEvent],
    atr: pd.Series,
    displacement_atr_multiple: float,
) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    ts = df["timestamp"]
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    atr_arr = atr.to_numpy()
    # structure_events are timestamped at their confirming bar's CLOSE, so the
    # lookup below must be keyed by close time too (see structure.py).
    ts_index = {t: i for i, t in enumerate(ts + TF_DURATION[timeframe])}

    for ev in structure_events:
        if ev.kind not in ("bos", "choch"):
            continue
        i = ts_index.get(ev.timestamp)
        if i is None:
            continue

        candle_range = h[i] - l[i]
        if pd.isna(atr_arr[i]) or candle_range < displacement_atr_multiple * atr_arr[i]:
            continue

        if ev.direction == "bullish":
            j = i - 1
            while j >= 0 and i - j <= _MAX_LOOKBACK and c[j] >= o[j]:
                j -= 1
            if j < 0 or i - j > _MAX_LOOKBACK:
                continue
            events.append(
                MarketEvent.zone(
                    kind="order_block_bullish",
                    timeframe=timeframe,
                    timestamp=ev.timestamp,
                    direction="bullish",
                    price_high=float(h[j]),
                    price_low=float(l[j]),
                    origin_candle_timestamp=ts.iloc[j],
                    structure_event=ev.kind,
                )
            )
        else:
            j = i - 1
            while j >= 0 and i - j <= _MAX_LOOKBACK and c[j] <= o[j]:
                j -= 1
            if j < 0 or i - j > _MAX_LOOKBACK:
                continue
            events.append(
                MarketEvent.zone(
                    kind="order_block_bearish",
                    timeframe=timeframe,
                    timestamp=ev.timestamp,
                    direction="bearish",
                    price_high=float(h[j]),
                    price_low=float(l[j]),
                    origin_candle_timestamp=ts.iloc[j],
                    structure_event=ev.kind,
                )
            )

    return events
