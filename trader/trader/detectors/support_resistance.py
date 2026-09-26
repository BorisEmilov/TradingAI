"""Classic horizontal support/resistance: a price cluster of >= min_touches
confirmed swing points. Causal: the level only becomes "known" at the
timestamp of its Nth touch, not retroactively at the first one -- you can't
call a level S/R until it's actually been respected more than once.
"""

from __future__ import annotations

from trader.events import MarketEvent


def detect_support_resistance(
    swings: list[MarketEvent], timeframe: str, tolerance_pct: float, min_touches: int
) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    pts = sorted(swings, key=lambda e: e.price)
    i = 0
    n = len(pts)

    while i < n:
        j = i
        anchor_price = pts[i].price
        group = [pts[i]]
        while j + 1 < n and abs(pts[j + 1].price - anchor_price) / anchor_price * 100.0 <= tolerance_pct:
            j += 1
            group.append(pts[j])

        if len(group) >= min_touches:
            by_time = sorted(group, key=lambda e: e.timestamp)
            confirm_ts = by_time[min_touches - 1].timestamp
            avg_price = sum(s.price for s in group) / len(group)
            events.append(
                MarketEvent.point(
                    kind="support_resistance",
                    timeframe=timeframe,
                    timestamp=confirm_ts,
                    direction="neutral",
                    price=float(avg_price),
                    touches=len(group),
                )
            )

        i = j + 1

    return events
