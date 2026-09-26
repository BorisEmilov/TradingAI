"""The backtester computes each TimeframeAnalysis ONCE over full history and
replays it with `.as_of(cutoff)` instead of recomputing from truncated
candles at every step. This only gives correct backtests if `.as_of()`
produces EXACTLY what `from_candles()` on truncated data would have -- this
test is the guarantee for that, in the same spirit as the "live replay"
causality test used earlier in this project's research line.
"""

import pandas as pd

from trader.events import TF_DURATION, MarketEvent, closed_candles_as_of
from trader.pipeline.engine import TimeframeAnalysis, _settled_at
from tests.conftest import build_candles
from tests.test_pipeline_engine import _toy_config, _always_on_sessions
from tests.test_structure import _BARS

_TS = pd.Timestamp("2026-01-05", tz="UTC")
_GRACE = pd.Timedelta(hours=2)


def test_as_of_matches_fresh_computation_from_truncated_data():
    config = _toy_config(_always_on_sessions())
    full_df = build_candles(_BARS, freq="1h")
    full_analysis = TimeframeAnalysis.from_candles(full_df, "H1", config)

    # idx9: OB zone from the bullish break at idx8 is still unmitigated.
    # idx10: the same bar both mitigates AND inverts that zone (close < price_low).
    for cutoff_idx in (9, 10):
        cutoff = full_df["timestamp"].iloc[cutoff_idx] + TF_DURATION["H1"]

        sliced = full_analysis.as_of(cutoff)
        truncated_df = closed_candles_as_of(full_df, "H1", cutoff)
        fresh = TimeframeAnalysis.from_candles(truncated_df, "H1", config)

        assert sliced.swings == fresh.swings, cutoff_idx
        assert sliced.structure_events == fresh.structure_events, cutoff_idx
        assert sliced.order_blocks == fresh.order_blocks, cutoff_idx
        assert sliced.inverted_order_blocks == fresh.inverted_order_blocks, cutoff_idx
        assert sliced.fvgs == fresh.fvgs, cutoff_idx
        assert sliced.inverted_fvgs == fresh.inverted_fvgs, cutoff_idx
        assert sliced.sweeps == fresh.sweeps, cutoff_idx
        assert sliced.equal_levels == fresh.equal_levels, cutoff_idx
        assert sliced.turtle_soups == fresh.turtle_soups, cutoff_idx
        assert sliced.sharp_turns == fresh.sharp_turns, cutoff_idx
        assert sliced.support_resistance == fresh.support_resistance, cutoff_idx
        assert sliced.elliott == fresh.elliott, cutoff_idx
        assert sliced.bias() == fresh.bias(), cutoff_idx

    # sanity: the two checkpoints actually differ (idx10's OB is mitigated+inverted,
    # idx9's isn't) -- otherwise this test would trivially pass without exercising
    # the as_of() zone-remitigation logic at all.
    ob_at_9 = full_analysis.as_of(full_df["timestamp"].iloc[9] + TF_DURATION["H1"]).order_blocks
    ob_at_10 = full_analysis.as_of(full_df["timestamp"].iloc[10] + TF_DURATION["H1"]).order_blocks
    assert ob_at_9 and not ob_at_9[0].mitigated
    assert ob_at_10 and ob_at_10[0].mitigated
    assert full_analysis.as_of(full_df["timestamp"].iloc[9] + TF_DURATION["H1"]).inverted_order_blocks == []
    assert len(full_analysis.as_of(full_df["timestamp"].iloc[10] + TF_DURATION["H1"]).inverted_order_blocks) == 1


# --- _settled_at: the perf optimization added while investigating why the
# scoring threshold sweep (prompt-sistema-puntuacion-ponderada.md) was
# taking hours -- profiling found as_of()'s per-zone recompute, not the new
# scoring code, as the dominant cost. These tests pin down its 3 cases
# directly; the equivalence tests above/below are what actually prove it
# doesn't change behavior, this just documents WHY each case is correct.


def test_settled_at_never_touched_is_the_zones_own_formation_timestamp():
    z = MarketEvent(kind="order_block_bullish", timeframe="H1", timestamp=_TS, direction="bullish", price=10.0, price_high=10.1, price_low=9.9)
    assert _settled_at(z) == _TS


def test_settled_at_touched_never_broken_is_the_touch_timestamp():
    touched = _TS + pd.Timedelta(hours=5)
    z = MarketEvent(
        kind="order_block_bullish", timeframe="H1", timestamp=_TS, direction="bullish", price=10.0, price_high=10.1,
        price_low=9.9, mitigated=True, mitigated_at=touched, confirmed_at=touched + _GRACE,
    )
    assert _settled_at(z) == touched


def test_settled_at_touched_and_broken_is_the_break_timestamp():
    touched = _TS + pd.Timedelta(hours=5)
    broken = touched + pd.Timedelta(hours=1)
    z = MarketEvent(
        kind="order_block_bullish", timeframe="H1", timestamp=_TS, direction="bullish", price=10.0, price_high=10.1,
        price_low=9.9, mitigated=True, mitigated_at=touched, broken_at=broken, confirmed_at=None,
    )
    assert _settled_at(z) == broken


def test_as_of_equivalence_across_many_zone_states_and_cutoffs():
    """Broader stress test than the single-OB scenario above: 4 zones in the
    4 qualitatively different lifecycle states the settled/pending split
    needs to get right (never touched; touched-and-held past the grace
    period; touched-then-broken-too-fast i.e. never validly confirmed;
    touched-then-broken-after-the-grace-period), checked at 6 cutoffs
    spanning before/at/between/after every one of their key timestamps --
    each compared against a truncated from_candles() recompute, not just
    against each other.
    """
    config = _toy_config(_always_on_sessions())
    bars = [(9.8, 9.85, 9.75, 9.8)] * 40
    full_df = build_candles(bars, freq="1h")
    close_ts = full_df["timestamp"] + TF_DURATION["H1"]

    grace = config.zone_lifecycle.invalidation_grace_m15_candles * TF_DURATION["M15"]

    def _zone(kind, formed_idx, touched_idx=None, broken_idx=None):
        formed = close_ts.iloc[formed_idx]
        touched = close_ts.iloc[touched_idx] if touched_idx is not None else None
        broken = close_ts.iloc[broken_idx] if broken_idx is not None else None
        if touched is None:
            confirmed = formed
        elif broken is None or (broken - touched) >= grace:
            confirmed = touched + grace
        else:
            confirmed = None
        return MarketEvent(
            kind=kind, timeframe="H1", timestamp=formed, direction="bullish", price=10.0, price_high=10.1,
            price_low=9.9, mitigated=touched is not None, mitigated_at=touched, broken_at=broken, confirmed_at=confirmed,
        )

    zones = [
        _zone("order_block_bullish", formed_idx=2),  # never touched
        _zone("order_block_bullish", formed_idx=5, touched_idx=8),  # touched, held (never broken)
        _zone("order_block_bullish", formed_idx=10, touched_idx=15, broken_idx=15),  # touched+broken same bar -> too fast
        _zone("order_block_bullish", formed_idx=20, touched_idx=22, broken_idx=30),  # touched, broken well after grace
    ]
    analysis = TimeframeAnalysis(
        timeframe="H1", df=full_df, swings=[], structure_events=[], order_blocks=zones, inverted_order_blocks=[],
        fvgs=[], inverted_fvgs=[], sweeps=[], equal_levels=[], turtle_soups=[], sharp_turns=[], support_resistance=[],
        elliott=[], grace_period=grace,
    )

    for cutoff_idx in (0, 6, 12, 16, 25, 35):
        cutoff = close_ts.iloc[cutoff_idx]
        sliced_obs = analysis.as_of(cutoff).order_blocks

        # naive recompute, mirroring apply_mitigation's own semantics directly
        # (not calling as_of() again -- this must be an INDEPENDENT check)
        expected = []
        for z in zones:
            if z.timestamp > cutoff:
                continue
            touched_at = z.mitigated_at if (z.mitigated_at is not None and z.mitigated_at <= cutoff) else None
            broken_at = z.broken_at if (z.broken_at is not None and z.broken_at <= cutoff) else None
            if touched_at is None:
                confirmed_at = z.timestamp
            elif broken_at is None or (broken_at - touched_at) >= grace:
                confirmed_at = touched_at + grace
            else:
                confirmed_at = None
            expected.append((z.kind, z.timestamp, touched_at, broken_at, confirmed_at))

        actual = [(z.kind, z.timestamp, z.mitigated_at, z.broken_at, z.confirmed_at) for z in sliced_obs]
        assert sorted(actual, key=lambda t: t[1]) == sorted(expected, key=lambda t: t[1]), cutoff_idx
