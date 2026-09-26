from __future__ import annotations

from dataclasses import dataclass

from trader.events import MarketEvent

FAMILY_BY_KIND = {
    "liquidity_sweep_bullish": "liquidity",
    "liquidity_sweep_bearish": "liquidity",
    "equal_highs": "liquidity",
    "equal_lows": "liquidity",
    "turtle_soup_bullish": "liquidity",
    "turtle_soup_bearish": "liquidity",
    "support_resistance": "support_resistance",
    "elliott_impulse_context": "elliott",
    "fair_value_gap_bullish": "fvg",
    "fair_value_gap_bearish": "fvg",
    "inverted_fair_value_gap_bullish": "fvg",
    "inverted_fair_value_gap_bearish": "fvg",
    "order_block_bullish": "order_block",
    "order_block_bearish": "order_block",
    "inverted_order_block_bullish": "order_block",
    "inverted_order_block_bearish": "order_block",
    "sharp_turn": "sharp_turn",
    "bos": "structure",
    "choch": "structure",
    "session": "session",
}


@dataclass(frozen=True)
class ConfluenceCheck:
    passed: bool
    families: set[str]
    timeframes: set[str]
    events: list[MarketEvent]


def evaluate_confluences(events: list[MarketEvent], min_confluences: int, min_timeframes: int) -> ConfluenceCheck:
    families = {FAMILY_BY_KIND[e.kind] for e in events if e.kind in FAMILY_BY_KIND}
    timeframes = {e.timeframe for e in events}
    passed = len(families) >= min_confluences and len(timeframes) >= min_timeframes
    return ConfluenceCheck(passed=passed, families=families, timeframes=timeframes, events=list(events))
