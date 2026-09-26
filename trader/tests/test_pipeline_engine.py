from dataclasses import replace

import pandas as pd

from trader.config import (
    ConfluenceConfig,
    ElliottConfig,
    FvgConfig,
    GatewayConfig,
    LiquidityConfig,
    M15ConfirmationConfig,
    NewsEventConfig,
    NewsFilterConfig,
    PoiConfig,
    RiskConfig,
    ScoringConfig,
    SessionsConfig,
    SessionWindowConfig,
    StructureConfig,
    SupportResistanceConfig,
    TraderConfig,
    ZoneLifecycleConfig,
)
from trader.events import TF_DURATION, MarketEvent
from trader.pipeline.engine import MultiTimeframePipeline, NoSignal, TimeframeAnalysis
from trader.signal import TradingSignal
from tests.conftest import build_candles

# Reuses the exact bar sequence hand-verified in test_structure.py: bars[0:10]
# contains a confirmed swing high (level 11.0, idx3), a confirmed swing low
# (level 9.5, idx6) that also serves as a valid bullish Order Block candle,
# and a displacement candle at idx8 whose close (11.4) breaks the swing high
# -> a single "bos" bullish event, with no later CHoCH inside the truncated
# window (that only appears at idx10, deliberately excluded).
_BARS = [
    (10.0, 10.2, 9.8, 10.0),
    (10.0, 10.5, 9.9, 10.3),
    (10.3, 11.0, 10.2, 10.8),
    (10.8, 10.6, 10.0, 10.1),
    (10.1, 10.2, 9.7, 9.9),
    (9.9, 10.0, 9.5, 9.6),
    (9.6, 9.9, 9.7, 9.8),
    (9.8, 10.6, 9.7, 10.5),
    (10.5, 11.5, 10.4, 11.4),
    (11.4, 11.6, 11.0, 11.1),
][:10]

_FLAT_BAR = (9.8, 9.85, 9.75, 9.8)


def _always_on_sessions() -> SessionsConfig:
    window = SessionWindowConfig(timezone="Etc/UTC", start_hour=0, end_hour=0)  # wraps -> full day
    return SessionsConfig(asia=window, london=window, new_york=window, killzone_london=window, killzone_new_york=window)


def _never_on_sessions() -> SessionsConfig:
    window = SessionWindowConfig(timezone="Etc/UTC", start_hour=2, end_hour=3)
    return SessionsConfig(asia=window, london=window, new_york=window, killzone_london=window, killzone_new_york=window)


def _toy_config(
    sessions: SessionsConfig,
    min_touches: int = 1,
    min_risk_atr_multiple: float = 0.0,
    m15_window_candles: int = 1,
    news_filter_enabled: bool = False,
    min_score_trend: float = 0.0,
    min_score_reversal: float = 0.0,
) -> TraderConfig:
    return TraderConfig(
        gateway=GatewayConfig(base_url="http://unused", timeout_seconds=1),
        symbols=["EURUSD"],
        structure=StructureConfig(swing_left_bars=1, swing_right_bars=1, displacement_atr_multiple=0.1, atr_period=3),
        liquidity=LiquidityConfig(equal_level_tolerance_pct=0.5, turtle_soup_lookback_bars=9, sweep_wick_min_pct=0.0),
        fvg=FvgConfig(min_gap_pct=0.0),
        support_resistance=SupportResistanceConfig(cluster_tolerance_pct=0.5, min_touches=min_touches),
        elliott=ElliottConfig(zigzag_deviation_pct=50.0),  # inert: too strict for this tiny dataset
        # min_risk_atr_multiple=0.0 (disabled) by default: most tests here are about
        # bias/POI/confluence/category logic, not the risk floor itself -- see
        # test_risk_below_minimum_floor_* below for that, which sets it explicitly.
        risk=RiskConfig(min_risk_reward=1.0, partial_at_progress_pct=0.5, min_risk_atr_multiple=min_risk_atr_multiple),
        confluence=ConfluenceConfig(min_confluences=3, min_timeframes=2),  # no longer read for gating, kept for compat
        sessions=sessions,
        poi=PoiConfig(tolerance_pct=0.1),
        zone_lifecycle=ZoneLifecycleConfig(invalidation_grace_m15_candles=6),
        # window_candles=1 (exact-only) by default so the existing toy scenarios --
        # written/verified against exact-match freshness -- keep behaving exactly as
        # before; test_m15_confirmation_window_rescues_a_stale_confirmation below sets
        # it explicitly. news_filter disabled by default: these toy timestamps are
        # arbitrary synthetic dates with no relationship to real NFP/FOMC calendar
        # dates, so leaving the real-world filter live here would make tests depend
        # on accidental calendar coincidences -- see test_news_filter.py instead.
        m15_confirmation=M15ConfirmationConfig(window_candles=m15_window_candles),
        news_filter=NewsFilterConfig(enabled=news_filter_enabled),
        # min_score_*=0.0 by default: these toy scenarios were written/verified
        # against the OLD confluence-count gate and were never designed with
        # composite-score fixtures in mind (D1 displacement/ATR ratio, Elliott,
        # S/R confluence, HTF sweep timing, etc. aren't deliberately tuned
        # here) -- a nonzero default would make unrelated tests (D1_bias,
        # session, POI, risk floor) fail or pass based on incidental score
        # values instead of the thing they actually test. The score gate
        # itself is covered by dedicated tests below and in test_scoring.py.
        scoring=ScoringConfig(min_score_trend=min_score_trend, min_score_reversal=min_score_reversal),
        signals_log_path="unused.jsonl",
    )


def _candles_by_tf(bars, m15_bars):
    return {
        "D1": build_candles(bars, freq="1D"),
        "H1": build_candles(bars, freq="1h"),
        "M15": build_candles(m15_bars, freq="15min"),
    }


def _turtle_soup_m15_bars():
    reversal = (9.7, 9.8, 9.6, 9.78)
    return [_FLAT_BAR] * 9 + [reversal]


def _mirror(bars, center):
    """Reflects OHLC bars around `center`: 2*center - price. A strictly
    decreasing bijection, so it preserves every ORDER relationship in
    reverse -- whatever bar was a swing high becomes a swing low, a BOS
    becomes bearish instead of bullish, etc, at the same indices/timing.
    Used to build a "mirror image" scenario without re-deriving prices by
    hand (h/l are swapped after negation so high >= low still holds).
    """
    return [(2 * center - o, 2 * center - l, 2 * center - h, 2 * center - c) for o, h, l, c in bars]


def test_insufficient_history_short_circuits_before_any_analysis():
    tiny = {tf: build_candles([_FLAT_BAR] * 3, freq="1h") for tf in ("D1", "H1", "M15")}
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions()))

    result = pipeline.run("EURUSD", tiny, as_of=pd.Timestamp("2027-01-01", tz="UTC"), current_price=9.8)

    assert isinstance(result, NoSignal)
    assert result.stage == "D1_bias"
    assert result.reason == "insufficient_history"


def test_no_bias_when_d1_is_flat():
    flat = {tf: build_candles([_FLAT_BAR] * 10, freq="1h") for tf in ("D1", "H1", "M15")}
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions()))

    result = pipeline.run("EURUSD", flat, as_of=pd.Timestamp("2027-01-01", tz="UTC"), current_price=9.8)

    assert isinstance(result, NoSignal)
    assert result.stage == "D1_bias"
    assert result.reason == "no_d1_bias_established"


def test_outside_active_sessions_blocks_before_poi_search():
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_never_on_sessions()))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, NoSignal)
    assert result.stage == "session"
    assert result.reason == "outside_active_sessions"


def test_no_h1_poi_when_price_far_from_any_zone():
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions()))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=500.0)

    assert isinstance(result, NoSignal)
    assert result.stage == "H1_poi"
    assert result.reason == "no_h1_poi_at_price"


def test_full_green_path_produces_a_valid_trading_signal():
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions()))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, TradingSignal), f"expected a signal, got {result!r}"
    assert result.direction == "long"
    assert result.category == "trend"  # D1's latest structure event here is a BOS, not a CHoCH
    assert result.bias_1d == "up"
    assert result.levels.risk_reward >= 1.0
    assert result.levels.sl < result.levels.entry < result.levels.tp
    assert len(result.confluences.families) >= 3
    assert len(result.confluences.timeframes) >= 2
    report = result.to_report()
    assert "Ratio R:R" in report


def test_risk_below_minimum_floor_is_rejected_not_widened():
    """Same exact scenario as the trend green path (risk_price=0.362,
    H1 ATR=0.867) but with the floor enabled at a multiple the structural
    stop can't clear (2x ATR = 1.73 > 0.362) -- must reject, not force the
    stop wider to comply.
    """
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions(), min_risk_atr_multiple=2.0))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, NoSignal)
    assert result.stage == "risk"
    assert result.reason == "risk_below_minimum_floor"


def test_risk_above_minimum_floor_still_passes():
    """Same scenario, floor enabled but at a multiple well below the actual
    risk (0.1x ATR = 0.087 < 0.362) -- should pass through exactly like the
    floor-disabled green path.
    """
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions(), min_risk_atr_multiple=0.1))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, TradingSignal), f"expected a signal, got {result!r}"


def test_reversal_via_fresh_choch_flips_direction_and_needs_more_confluences():
    """D1 uses the FULL 11-bar sequence (not truncated to 10, unlike the
    trend test) -- its latest structure event is the CHoCH at idx10
    (bearish), not the BOS at idx8, so bias()="down" with a fresh reversal,
    not an established trend. H1/M15 are mirrored (bearish instead of
    bullish) so their POI/confirmation match that bearish direction. This
    should route through the "reversal" category (min_confluences_reversal,
    stricter than trend's).
    """
    from tests.test_structure import _BARS as _full_11_bars  # the untruncated sequence, with the CHoCH at idx10

    center = 10.5
    d1_bars = _full_11_bars  # unmirrored: latest event is already the bearish CHoCH at idx10
    h1_bars = _mirror(_BARS[:10], center)  # mirrored/truncated: bearish OB, matches the CHoCH direction
    m15_bars = _mirror(_turtle_soup_m15_bars(), center=9.8)  # mirrored: bearish confirmation

    candles = {
        "D1": build_candles(d1_bars, freq="1D"),
        "H1": build_candles(h1_bars, freq="1h"),
        "M15": build_candles(m15_bars, freq="15min"),
    }
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    current_price = 2 * center - 9.8  # mirror of the trend test's current_price=9.8 around the H1 mirror center
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions()))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=current_price)

    assert isinstance(result, TradingSignal), f"expected a signal, got {result!r}"
    assert result.direction == "short"
    assert result.category == "reversal"
    assert result.bias_1d == "down"
    assert result.levels.risk_reward >= 1.0
    assert result.levels.tp < result.levels.entry < result.levels.sl
    assert len(result.confluences.families) >= 4  # stricter minimum for reversal trades


def test_d1_reversal_trigger_requires_confirmation_after_the_sweep():
    """Unit-level check of the sweep+exhaustion trigger (Trigger A): a
    bearish sweep with no turtle-soup/sharp-turn confirmation afterward is
    not a live trigger; adding one after the sweep's timestamp makes it one.
    """
    ts0 = pd.Timestamp("2026-01-05", tz="UTC")
    empty_df = build_candles([], freq="1D")

    def _analysis(sweeps, turtle_soups, sharp_turns):
        return TimeframeAnalysis(
            timeframe="D1", df=empty_df, swings=[], structure_events=[], order_blocks=[], inverted_order_blocks=[],
            fvgs=[], inverted_fvgs=[], sweeps=sweeps, equal_levels=[], turtle_soups=turtle_soups,
            sharp_turns=sharp_turns, support_resistance=[], elliott=[],
        )

    sweep = MarketEvent.point("liquidity_sweep_bearish", "D1", ts0, "bearish", 10.0)

    no_confirmation = _analysis([sweep], [], [])
    assert MultiTimeframePipeline._d1_reversal_trigger(no_confirmation, "bearish") is None

    late_confirmation = MarketEvent.point("sharp_turn", "D1", ts0 + pd.Timedelta(days=1), "bearish", 9.9)
    with_confirmation = _analysis([sweep], [], [late_confirmation])
    trigger = MultiTimeframePipeline._d1_reversal_trigger(with_confirmation, "bearish")
    assert trigger is late_confirmation

    wrong_direction_confirmation = MarketEvent.point("sharp_turn", "D1", ts0 + pd.Timedelta(days=1), "bullish", 9.9)
    still_none = _analysis([sweep], [], [wrong_direction_confirmation])
    assert MultiTimeframePipeline._d1_reversal_trigger(still_none, "bearish") is None


def _turtle_soup_m15_bars_stale_by_one():
    """Same reversal as `_turtle_soup_m15_bars`, with one more flat bar
    appended -- the confirmation is now 1 candle (15 min) old by the time the
    M15 series "closes", exactly the gap `window_candles=2` is meant to
    rescue (prompt-implementar-m15-n2.md).
    """
    return _turtle_soup_m15_bars() + [_FLAT_BAR]


def test_m15_confirmation_exact_window_rejects_a_one_candle_stale_confirmation():
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars_stale_by_one())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions(), m15_window_candles=1))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, NoSignal)
    assert result.stage == "M15_confirmation"
    assert result.reason == "no_m15_confirmation"


def test_m15_confirmation_window_2_rescues_the_same_stale_confirmation():
    """Identical scenario to the rejection test above, only `window_candles`
    differs -- proves the relaxation actually fires end-to-end, not just at
    the `_latest_confirmation` unit level.
    """
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars_stale_by_one())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions(), m15_window_candles=2))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, TradingSignal), f"expected a signal, got {result!r}"


def test_latest_confirmation_uses_the_analysis_own_timeframe_not_hardcoded_m15():
    """Regression guard for prompt-experimento-d1-30m-15m.md: `_latest_confirmation`
    and `_candidate_confirmation_timestamps` used to hardcode `TF_DURATION["M15"]`
    for the freshness-window math, silently correct only because every
    production caller happened to pass an M15 analysis. Builds an M30
    analysis instead and confirms the exact-vs-stale-by-one-candle behavior
    uses 30-minute steps, not 15-minute ones.
    """
    reversal = (9.7, 9.8, 9.6, 9.78)
    bars = [_FLAT_BAR] * 9 + [reversal]
    df = build_candles(bars, freq="30min")
    confirmation_ts = df["timestamp"].iloc[9] + TF_DURATION["M30"]
    event = MarketEvent.point("turtle_soup_bullish", "M30", confirmation_ts, "bullish", 9.78)
    analysis = TimeframeAnalysis(
        timeframe="M30", df=df, swings=[], structure_events=[], order_blocks=[], inverted_order_blocks=[],
        fvgs=[], inverted_fvgs=[], sweeps=[], equal_levels=[], turtle_soups=[event], sharp_turns=[],
        support_resistance=[], elliott=[],
    )

    exact = MultiTimeframePipeline._latest_confirmation(analysis, "bullish", window_candles=1)
    assert exact is event

    # one M30 candle (30 min) later: window=1 must NOT find it via a wrong
    # 15-min step, window=2 must find it via the correct 30-min step.
    stale_df = build_candles(bars + [_FLAT_BAR], freq="30min")
    stale_analysis = TimeframeAnalysis(
        timeframe="M30", df=stale_df, swings=[], structure_events=[], order_blocks=[], inverted_order_blocks=[],
        fvgs=[], inverted_fvgs=[], sweeps=[], equal_levels=[], turtle_soups=[event], sharp_turns=[],
        support_resistance=[], elliott=[],
    )
    assert MultiTimeframePipeline._latest_confirmation(stale_analysis, "bullish", window_candles=1) is None
    assert MultiTimeframePipeline._latest_confirmation(stale_analysis, "bullish", window_candles=2) is event


def test_news_filter_blocks_before_poi_search():
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    config = _toy_config(_always_on_sessions(), news_filter_enabled=True)
    config = replace(
        config,
        news_filter=replace(
            config.news_filter,
            nfp_auto=False,
            manual_events=[NewsEventConfig(timestamp_utc=as_of.isoformat(), label="TEST_EVENT")],
        ),
    )
    pipeline = MultiTimeframePipeline(config)

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, NoSignal)
    assert result.stage == "news_filter"
    assert result.reason == "high_impact_news_window"


def test_news_filter_disabled_by_default_does_not_block():
    # Sanity check that pairs with the test above: the exact same scenario
    # that blocks when news_filter is enabled still reaches a signal when
    # it's off (the `_toy_config` default) -- proves the block above is
    # really coming from the news filter, not something else in the fixture.
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions()))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, TradingSignal), f"expected a signal, got {result!r}"


def test_score_below_minimum_blocks_signal_not_confluence_count():
    """The green-path scenario passes today with min_score_trend=0 (the
    _toy_config default) -- this proves the SAME scenario is rejected, at the
    "score" stage specifically (not "confluence", which no longer exists as a
    gate), once the threshold is set above whatever this fixture actually
    scores. Companion to test_score_at_minimum_still_passes below.
    """
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]
    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions(), min_score_trend=1000.0))

    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, NoSignal)
    assert result.stage == "score"
    assert result.reason == "score_below_minimum"


def test_score_at_minimum_still_passes():
    """Reads back the green path's actual score, then re-runs with the
    threshold set exactly at that value -- proves the gate is a `>=` (score
    equal to the minimum passes), not a strict `>` that would silently
    reject the boundary case.
    """
    candles = _candles_by_tf(_BARS, _turtle_soup_m15_bars())
    as_of = candles["D1"]["timestamp"].iloc[-1] + TF_DURATION["D1"]

    baseline = MultiTimeframePipeline(_toy_config(_always_on_sessions()))
    baseline_result = baseline.run("EURUSD", candles, as_of=as_of, current_price=9.8)
    assert isinstance(baseline_result, TradingSignal)
    actual_score = baseline_result.score.total

    pipeline = MultiTimeframePipeline(_toy_config(_always_on_sessions(), min_score_trend=actual_score))
    result = pipeline.run("EURUSD", candles, as_of=as_of, current_price=9.8)

    assert isinstance(result, TradingSignal), f"expected a signal, got {result!r}"
    assert result.score.total == actual_score
