import dataclasses

import pandas as pd

from tests.conftest import build_candles
from trader.dominant_reason import (
    DOMINANT_REASON_THRESHOLD,
    DominantReasonEngine,
    _displacement_ratio_at,
    _matching_level_touches,
    _wick_ratio_at,
)
from trader.events import MarketEvent


def _ts_to_idx(df: pd.DataFrame) -> dict:
    return {ts: i for i, ts in enumerate(df["timestamp"])}


def test_wick_ratio_bearish_sweep_uses_upper_wick():
    df = build_candles([(10, 12, 9, 10)])
    idx_map = _ts_to_idx(df)
    event = MarketEvent.point("liquidity_sweep", "H1", df["timestamp"].iloc[0], "bearish", 12.0)
    ratio = _wick_ratio_at(df, idx_map, event)
    assert abs(ratio - (12 - 10) / (12 - 9)) < 1e-9


def test_wick_ratio_bullish_sweep_uses_lower_wick():
    df = build_candles([(10, 11, 8, 10)])
    idx_map = _ts_to_idx(df)
    event = MarketEvent.point("liquidity_sweep", "H1", df["timestamp"].iloc[0], "bullish", 8.0)
    ratio = _wick_ratio_at(df, idx_map, event)
    assert abs(ratio - (10 - 8) / (11 - 8)) < 1e-9


def test_wick_ratio_missing_timestamp_returns_zero():
    df = build_candles([(10, 11, 9, 10)])
    idx_map = _ts_to_idx(df)
    other_ts = df["timestamp"].iloc[0] + pd.Timedelta(hours=99)
    event = MarketEvent.point("liquidity_sweep", "H1", other_ts, "bearish", 11.0)
    assert _wick_ratio_at(df, idx_map, event) == 0.0


def test_matching_level_touches_price_proximity():
    sweep = MarketEvent.point("liquidity_sweep", "H1", pd.Timestamp("2026-01-05", tz="UTC"), "bearish", 100.0)
    close_level = MarketEvent.point(
        "equal_highs", "H1", pd.Timestamp("2026-01-04", tz="UTC"), "neutral", 100.03, count=5
    )
    far_level = MarketEvent.point(
        "equal_highs", "H1", pd.Timestamp("2026-01-04", tz="UTC"), "neutral", 95.0, count=9
    )
    assert _matching_level_touches(sweep, [far_level, close_level]) == 5


def test_matching_level_touches_defaults_to_one_when_no_match():
    sweep = MarketEvent.point("liquidity_sweep", "H1", pd.Timestamp("2026-01-05", tz="UTC"), "bearish", 100.0)
    far_level = MarketEvent.point(
        "equal_highs", "H1", pd.Timestamp("2026-01-04", tz="UTC"), "neutral", 95.0, count=9
    )
    assert _matching_level_touches(sweep, [far_level]) == 1


def test_matching_level_touches_ignores_future_levels():
    sweep = MarketEvent.point("liquidity_sweep", "H1", pd.Timestamp("2026-01-05", tz="UTC"), "bearish", 100.0)
    future_level = MarketEvent.point(
        "equal_highs", "H1", pd.Timestamp("2026-01-06", tz="UTC"), "neutral", 100.01, count=7
    )
    assert _matching_level_touches(sweep, [future_level]) == 1


def test_displacement_ratio_basic():
    df = build_candles([(10, 15, 9, 14)])
    idx_map = _ts_to_idx(df)
    atr = pd.Series([2.0])
    ratio = _displacement_ratio_at(df, atr, idx_map, df["timestamp"].iloc[0])
    assert abs(ratio - (15 - 9) / 2.0) < 1e-9


def test_displacement_ratio_zero_atr_is_safe():
    df = build_candles([(10, 15, 9, 14)])
    idx_map = _ts_to_idx(df)
    atr = pd.Series([0.0])
    assert _displacement_ratio_at(df, atr, idx_map, df["timestamp"].iloc[0]) == 0.0


def _fill_to_min_sample(engine: DominantReasonEngine, direction: str, n: int, kind: str, base_ts: pd.Timestamp):
    events = []
    for i in range(n):
        ts = base_ts + pd.Timedelta(hours=i)
        ev = MarketEvent.point(kind, "H1", ts, direction, 1.0 + i)
        events.append(ev)
        if kind == "liquidity_sweep":
            engine.ingest_sweep(ev, wick_ratio=1.0, touches=i)  # raw = i
        elif kind == "order_block_bullish" or kind == "order_block_bearish":
            engine.ingest_zone(ev, displacement_ratio=float(i), fresh=False)
        elif kind == "equal_highs" or kind == "equal_lows":
            engine.ingest_level(ev, touches=float(i), direction=direction)
    return events


def test_best_candidate_picks_max_not_sum_and_enforces_threshold():
    engine = DominantReasonEngine()
    base_ts = pd.Timestamp("2026-01-01", tz="UTC")

    # Fill 30 filler events per tracker (raw 0..29) so the 31st insertion is evaluable.
    _fill_to_min_sample(engine, "bullish", 30, "liquidity_sweep", base_ts)
    _fill_to_min_sample(engine, "bullish", 30, "order_block_bullish", base_ts + pd.Timedelta(days=1))
    _fill_to_min_sample(engine, "bullish", 30, "equal_highs", base_ts + pd.Timedelta(days=2))

    sweep = MarketEvent.point("liquidity_sweep", "H1", base_ts + pd.Timedelta(hours=100), "bullish", 1.0)
    engine.ingest_sweep(sweep, wick_ratio=1.0, touches=100)  # raw=100 -> top of 30-sample history -> 100th pct

    zone = MarketEvent.point("order_block_bullish", "H1", base_ts + pd.Timedelta(hours=101), "bullish", 1.0)
    engine.ingest_zone(zone, displacement_ratio=15.0, fresh=False)  # raw=15 -> mid of 0..29 -> 50th pct

    level = MarketEvent.point("equal_highs", "H1", base_ts + pd.Timedelta(hours=102), "bullish", 1.0)
    engine.ingest_level(level, touches=29.5, direction="bullish")  # near-top -> ~96.6th pct

    best = engine.best_candidate("bullish", current_price=1.0, recent_sweep=sweep, overlapping_zone=zone, nearby_level=level)
    assert best is not None
    assert best.kind == "liquidity_taken"
    assert best.strength_percentile == 100.0

    # Now only the sub-threshold zone candidate is available -> None.
    below_threshold = engine.best_candidate("bullish", current_price=1.0, recent_sweep=None, overlapping_zone=zone, nearby_level=None)
    assert below_threshold is None
    assert DOMINANT_REASON_THRESHOLD == 80.0


def test_best_candidate_none_when_no_percentile_computed_yet():
    engine = DominantReasonEngine()
    sweep = MarketEvent.point("liquidity_sweep", "H1", pd.Timestamp("2026-01-01", tz="UTC"), "bullish", 1.0)
    engine.ingest_sweep(sweep, wick_ratio=1.0, touches=1)  # below min_sample -> no percentile cached
    assert engine.best_candidate("bullish", 1.0, recent_sweep=sweep, overlapping_zone=None, nearby_level=None) is None


def test_causality_future_events_dont_change_earlier_candidate():
    base_ts = pd.Timestamp("2026-01-01", tz="UTC")

    engine_a = DominantReasonEngine()
    _fill_to_min_sample(engine_a, "bearish", 30, "liquidity_sweep", base_ts)
    early_sweep = MarketEvent.point("liquidity_sweep", "H1", base_ts + pd.Timedelta(hours=100), "bearish", 1.0)
    engine_a.ingest_sweep(early_sweep, wick_ratio=1.0, touches=50)
    result_a = engine_a.best_candidate("bearish", 1.0, recent_sweep=early_sweep, overlapping_zone=None, nearby_level=None)

    engine_b = DominantReasonEngine()
    _fill_to_min_sample(engine_b, "bearish", 30, "liquidity_sweep", base_ts)
    engine_b.ingest_sweep(early_sweep, wick_ratio=1.0, touches=50)
    result_b_before_future = engine_b.best_candidate("bearish", 1.0, recent_sweep=early_sweep, overlapping_zone=None, nearby_level=None)
    # Ingest wild future sweeps AFTER the fact.
    for i in range(20):
        future_ev = MarketEvent.point(
            "liquidity_sweep", "H1", base_ts + pd.Timedelta(hours=200 + i), "bearish", 1.0
        )
        engine_b.ingest_sweep(future_ev, wick_ratio=1.0, touches=9999)
    result_b_after_future = engine_b.best_candidate("bearish", 1.0, recent_sweep=early_sweep, overlapping_zone=None, nearby_level=None)

    assert result_a == result_b_before_future == result_b_after_future


def test_key_level_decay_reduces_percentile_by_age():
    engine = DominantReasonEngine()
    base_ts = pd.Timestamp("2026-01-01", tz="UTC")
    _fill_to_min_sample(engine, "bullish", 30, "equal_highs", base_ts)

    level = MarketEvent.point("equal_highs", "H1", base_ts + pd.Timedelta(hours=100), "bullish", 1.0)
    engine.ingest_level(level, touches=100.0, direction="bullish")  # raw percentile -> 100.0

    no_decay = engine.best_candidate("bullish", 1.0, recent_sweep=None, overlapping_zone=None, nearby_level=level)
    assert no_decay is not None
    assert no_decay.strength_percentile == 100.0

    decayed_small_age = engine.best_candidate(
        "bullish", 1.0, recent_sweep=None, overlapping_zone=None, nearby_level=level, nearby_level_age_days=10.0
    )
    assert decayed_small_age is not None
    assert abs(decayed_small_age.strength_percentile - 100.0 / (1.0 + 10.0 / 180.0)) < 1e-9

    decayed_at_halflife = engine.best_candidate(
        "bullish", 1.0, recent_sweep=None, overlapping_zone=None, nearby_level=level, nearby_level_age_days=180.0
    )
    assert decayed_at_halflife is None  # 100 decays to 50 at the 180-day half-life -- below the 80 threshold


def test_zone_candidate_survives_as_of_replace_copy():
    """Regression test: TimeframeAnalysis.as_of() returns a
    dataclasses.replace()-built COPY of a zone event (different id(), same
    content) whenever its causally-truncated confirmed_at/broken_at differ
    from the full-history values. The engine must key its cache by content,
    not object identity, or a zone looked up through such a copy would
    silently miss its already-computed percentile."""
    engine = DominantReasonEngine()
    base_ts = pd.Timestamp("2026-01-01", tz="UTC")
    _fill_to_min_sample(engine, "bullish", 30, "order_block_bullish", base_ts)

    original_zone = MarketEvent.point("order_block_bullish", "H1", base_ts + pd.Timedelta(hours=200), "bullish", 1.0)
    engine.ingest_zone(original_zone, displacement_ratio=100.0, fresh=False)

    # Simulate what as_of() does: a content-identical but distinct object.
    copied_zone = dataclasses.replace(original_zone, mitigated=True, mitigated_at=base_ts + pd.Timedelta(hours=201))
    assert copied_zone is not original_zone
    assert id(copied_zone) != id(original_zone)

    result = engine.best_candidate("bullish", 1.0, recent_sweep=None, overlapping_zone=copied_zone, nearby_level=None)
    assert result is not None
    assert result.kind == "htf_zone"
    assert result.strength_percentile == 100.0
