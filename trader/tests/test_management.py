import pandas as pd

from tests.conftest import build_candles
from trader.backtest.costs import SymbolCost
from trader.events import MarketEvent
from trader.management import (
    CONVICTION_MAX,
    CONVICTION_MIN,
    conviction_multiplier,
    liquidity_targets,
    simulate_managed_trade,
    structural_buffer,
)
from trader.pipeline.engine import TimeframeAnalysis

ZERO_COST = SymbolCost(symbol="TEST", point=0.0001, avg_spread_points=0.0, avg_spread_price=0.0)


def _analysis(df, structure_events=None, support_resistance=None, equal_levels=None, swings=None) -> TimeframeAnalysis:
    return TimeframeAnalysis(
        timeframe="M15",
        df=df,
        swings=swings or [],
        structure_events=structure_events or [],
        order_blocks=[],
        inverted_order_blocks=[],
        fvgs=[],
        inverted_fvgs=[],
        sweeps=[],
        equal_levels=equal_levels or [],
        turtle_soups=[],
        sharp_turns=[],
        support_resistance=support_resistance or [],
        elliott=[],
    )


def test_structural_buffer_uses_last_20_bar_mean_range():
    bars = [(10, 10 + i * 0.1, 10, 10) for i in range(1, 25)]  # ranges 0.1..2.4
    df = build_candles(bars, freq="15min")
    expected_mean_range = (df["high"] - df["low"]).tail(20).mean()
    assert abs(structural_buffer(df) - expected_mean_range * 0.1) < 1e-9


def test_structural_buffer_empty_df_is_safe():
    df = build_candles([], freq="15min")
    assert structural_buffer(df) == 0.0001


def test_liquidity_targets_sorted_nearest_first_long():
    df = build_candles([(1, 1, 1, 1)], freq="15min")
    levels = [
        MarketEvent.point("support_resistance", "M15", df["timestamp"].iloc[0], "neutral", 110.0),
        MarketEvent.point("support_resistance", "M15", df["timestamp"].iloc[0], "neutral", 105.0),
        MarketEvent.point("support_resistance", "M15", df["timestamp"].iloc[0], "neutral", 95.0),  # below entry, excluded
    ]
    analysis = _analysis(df, support_resistance=levels)
    targets = liquidity_targets("long", analysis, entry=100.0)
    assert targets == [105.0, 110.0]


def test_liquidity_targets_sorted_nearest_first_short():
    df = build_candles([(1, 1, 1, 1)], freq="15min")
    levels = [
        MarketEvent.point("support_resistance", "M15", df["timestamp"].iloc[0], "neutral", 90.0),
        MarketEvent.point("support_resistance", "M15", df["timestamp"].iloc[0], "neutral", 95.0),
        MarketEvent.point("support_resistance", "M15", df["timestamp"].iloc[0], "neutral", 105.0),  # above entry, excluded
    ]
    analysis = _analysis(df, support_resistance=levels)
    targets = liquidity_targets("short", analysis, entry=100.0)
    assert targets == [95.0, 90.0]


def test_conviction_multiplier_at_thresholds_is_one():
    assert conviction_multiplier(80.0, 70.0) == 1.0


def test_conviction_multiplier_clamped_at_max():
    assert conviction_multiplier(100.0, 100.0) == CONVICTION_MAX


def test_conviction_multiplier_clamped_at_min():
    assert conviction_multiplier(0.0, 0.0) == CONVICTION_MIN


def _base_bars(n: int, price: float = 100.0):
    return [(price, price + 0.5, price - 0.5, price) for _ in range(n)]


def test_simple_stop_loss_hit():
    bars = _base_bars(3) + [(100, 100.2, 97.0, 97.5)]  # bar after fill dives through SL=98
    df = build_candles(bars, freq="15min")
    analysis = _analysis(df)
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=None,
        final_target=200.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is not None
    assert result.exit_reason == "stop_loss"
    assert abs(result.gross_r - (-1.0)) < 1e-9
    assert result.sl_ratchets == 0


def test_sl_ratchets_on_favorable_bos_and_exits_at_new_level():
    fill_bar = (100, 100.5, 99.5, 100)
    bos_bar = (100, 103.0, 100, 103.0)  # closes above a swing high -> bos event we'll inject at this bar's close
    pullback_bar = (103, 103.2, 101.4, 101.5)  # pulls back but should NOT hit original SL=98
    df = build_candles([fill_bar, fill_bar, bos_bar, pullback_bar], freq="15min")
    bos_close_ts = df["timestamp"].iloc[2] + pd.Timedelta(minutes=15)
    bos_event = MarketEvent.point(
        "bos", "M15", bos_close_ts, "bullish", 103.0, broken_level=102.0
    )
    analysis = _analysis(df, structure_events=[bos_event])
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=None,
        final_target=500.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is not None
    assert result.sl_ratchets == 1
    # buffer = mean(high-low over available bars)*0.1; new sl should be well above original 98
    assert result.final_sl > 98.0
    assert result.exit_reason in ("trailing_stop", "max_holding_period")


def test_partial_then_structural_invalidation_exit():
    fill_bar = (100, 100.5, 99.5, 100)
    partial_bar = (100, 106.0, 100, 105.5)  # hits partial target of 105
    invalidation_bar = (105, 105.2, 103.0, 103.5)  # opposite BOS lands here
    df = build_candles([fill_bar, fill_bar, partial_bar, invalidation_bar], freq="15min")
    inval_close_ts = df["timestamp"].iloc[3] + pd.Timedelta(minutes=15)
    opposite_bos = MarketEvent.point("choch", "M15", inval_close_ts, "bearish", 103.5, broken_level=104.0)
    analysis = _analysis(df, structure_events=[opposite_bos])
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=105.0,
        final_target=200.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is not None
    assert result.partial_taken is True
    assert result.exit_reason == "structural_invalidation_post_partial"


def test_opposite_bos_before_partial_invalidates():
    fill_bar = (100, 100.5, 99.5, 100)
    invalidation_bar = (100, 100.2, 98.5, 98.7)  # doesn't hit SL=98, but opposite BOS fires here
    df = build_candles([fill_bar, fill_bar, invalidation_bar], freq="15min")
    inval_close_ts = df["timestamp"].iloc[2] + pd.Timedelta(minutes=15)
    opposite_bos = MarketEvent.point("choch", "M15", inval_close_ts, "bearish", 98.7, broken_level=99.0)
    analysis = _analysis(df, structure_events=[opposite_bos])
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=110.0,
        final_target=200.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is not None
    assert result.partial_taken is False
    assert result.exit_reason == "structural_invalidation_pre_partial"


def test_max_holding_period_tail_exit():
    bars = _base_bars(2) + [(100, 100.3, 99.7, 100.1) for _ in range(400)]  # 400 bars * 15min > 3 days
    df = build_candles(bars, freq="15min")
    analysis = _analysis(df)
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="range", dominant_reason_kind="htf_zone",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=None,
        final_target=200.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is not None
    assert result.exit_reason == "max_holding_period"


def test_risk_pct_applied_uses_conviction_multiplier():
    bars = _base_bars(3) + [(100, 100.2, 97.0, 97.5)]
    df = build_candles(bars, freq="15min")
    analysis = _analysis(df)
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=None,
        final_target=200.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.8,
    )
    assert result is not None
    assert abs(result.risk_pct_applied - 0.9) < 1e-9


def test_final_target_hit_without_partial():
    fill_bar = (100, 100.5, 99.5, 100)
    tp_bar = (100, 111.0, 100, 110.5)  # runs straight to final_target=110, no intermediate level
    df = build_candles([fill_bar, fill_bar, tp_bar], freq="15min")
    analysis = _analysis(df)
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=None,
        final_target=110.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is not None
    assert result.exit_reason == "take_profit"
    assert result.partial_taken is False
    assert abs(result.gross_r - ((110.0 - 100.0) / (100.0 - 98.0))) < 1e-9


def test_final_target_hit_after_partial():
    fill_bar = (100, 100.5, 99.5, 100)
    partial_bar = (100, 106.0, 100, 105.5)  # hits partial target of 105
    tp_bar = (105.5, 111.0, 105, 110.5)  # later reaches final_target=110
    df = build_candles([fill_bar, fill_bar, partial_bar, tp_bar], freq="15min")
    analysis = _analysis(df)
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=105.0,
        final_target=110.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is not None
    assert result.exit_reason == "take_profit_post_partial"
    assert result.partial_taken is True
    partial_r = (105.0 - 100.0) / (100.0 - 98.0)
    tp_r = (110.0 - 100.0) / (100.0 - 98.0)
    assert abs(result.gross_r - (0.5 * partial_r + 0.5 * tp_r)) < 1e-9


def test_reward_below_zero_at_fill_returns_none():
    # final_target below fill_price for a long -> reward_price <= 0 -> reject
    bars = [(100, 100.5, 99.5, 100), (100, 100.5, 99.5, 100)]
    df = build_candles(bars, freq="15min")
    analysis = _analysis(df)
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=None,
        final_target=99.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is None


def test_invalid_setup_at_fill_returns_none():
    # fill_price below original_sl for a long -> risk_price <= 0 -> reject
    bars = [(100, 100.5, 99.5, 100), (90, 90.5, 89.5, 90)]
    df = build_candles(bars, freq="15min")
    analysis = _analysis(df)
    result = simulate_managed_trade(
        symbol="TEST", direction="long", regime="trend_up", dominant_reason_kind="liquidity_taken",
        signal_bar_idx=0, m15_df=df, m15_analysis=analysis, original_sl=98.0, partial_target=None,
        final_target=200.0, symbol_cost=ZERO_COST, base_risk_pct=0.5, conviction_mult=1.0,
    )
    assert result is None
