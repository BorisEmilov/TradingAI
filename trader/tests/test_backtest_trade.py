import pandas as pd

from trader.backtest.costs import SymbolCost
from trader.backtest.trade import simulate_trade
from trader.events import MarketEvent
from trader.pipeline.confluence import ConfluenceCheck
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.scoring import ScoreBreakdown
from trader.risk.levels import TradeLevels
from trader.sessions import SessionState
from trader.signal import TradingSignal
from tests.conftest import build_candles

_ZERO_COST = SymbolCost(symbol="EURUSD", point=0.0001, avg_spread_points=0.0, avg_spread_price=0.0)


def _empty_h1_analysis() -> TimeframeAnalysis:
    return TimeframeAnalysis(
        timeframe="H1", df=build_candles([], freq="1h"),
        swings=[], structure_events=[], order_blocks=[], inverted_order_blocks=[],
        fvgs=[], inverted_fvgs=[], sweeps=[], equal_levels=[], turtle_soups=[],
        sharp_turns=[], support_resistance=[], elliott=[],
    )


def _h1_analysis_with_choch(ts: pd.Timestamp, direction: str) -> TimeframeAnalysis:
    a = _empty_h1_analysis()
    a.structure_events.append(MarketEvent.point("choch", "H1", ts, direction, 100.0))
    return a


def _signal(entry: float, sl: float, tp: float, generated_at: pd.Timestamp, direction: str = "long") -> TradingSignal:
    session = SessionState(timestamp=generated_at, asia=False, london=True, new_york=False, killzone_london=True, killzone_new_york=False)
    poi = MarketEvent.zone("order_block_bullish", "H1", generated_at, "bullish", price_high=entry, price_low=sl)
    confirmation = MarketEvent.point("choch", "M15", generated_at, "bullish", entry)
    confluences = ConfluenceCheck(passed=True, families={"structure", "order_block", "liquidity"}, timeframes={"H1", "M15"}, events=[])
    score = ScoreBreakdown(bias=0.0, poi=0.0, confirmation=0.0, sweep=0.0, session=0.0, diversity=0.0)
    levels = TradeLevels(direction=direction, entry=entry, sl=sl, tp=tp, risk_reward=abs(tp - entry) / abs(entry - sl))
    return TradingSignal(
        symbol="EURUSD", direction=direction, category="trend", bias_1d="up", session=session, poi=poi,
        confirmation=confirmation, confluences=confluences, score=score, levels=levels, partial_at_progress_pct=0.5,
        generated_at=generated_at,
    )


def test_stop_loss_hit_before_partial_is_exactly_minus_one_r():
    bars = [
        (100.0, 100.1, 99.9, 100.0),  # signal bar (unused directly)
        (100.0, 100.1, 99.5, 99.8),  # fill bar: open=100 (fill price)
        (99.8, 99.9, 98.9, 99.0),  # low dips through SL (99.0)
    ]
    df = build_candles(bars, freq="15min")
    signal = _signal(entry=100.0, sl=99.0, tp=103.0, generated_at=df["timestamp"].iloc[0])

    trade = simulate_trade(signal, df, signal_bar_idx=0, h1_analysis=_empty_h1_analysis(), symbol_cost=_ZERO_COST, dominant_confluence_family="structure")

    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.entry == 100.0
    assert trade.gross_r == -1.0
    assert trade.net_r == -1.0
    assert trade.partial_taken is False


def test_partial_then_take_profit_weighted_r():
    bars = [
        (100.0, 100.1, 99.9, 100.0),  # signal bar
        (100.0, 100.2, 99.8, 100.1),  # fill bar: open=100
        (100.1, 101.6, 99.9, 101.5),  # reaches partial level (101.5 = 100 + 0.5*(103-100))
        (101.5, 103.2, 101.0, 103.1),  # reaches TP (103.0)
    ]
    df = build_candles(bars, freq="15min")
    signal = _signal(entry=100.0, sl=99.0, tp=103.0, generated_at=df["timestamp"].iloc[0])

    trade = simulate_trade(signal, df, signal_bar_idx=0, h1_analysis=_empty_h1_analysis(), symbol_cost=_ZERO_COST, dominant_confluence_family="liquidity")

    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.partial_taken is True
    # partial_r = (101.5-100)/(100-99) = 1.5; tp_r = (103-100)/(100-99) = 3.0
    # net_r = 0.5*1.5 + 0.5*3.0 = 2.25
    assert trade.gross_r == 2.25
    assert trade.net_r == 2.25
    assert trade.exit_bar_index == 3


def test_partial_then_breakeven_stop():
    bars = [
        (100.0, 100.1, 99.9, 100.0),
        (100.0, 100.2, 99.8, 100.1),
        (100.1, 101.6, 99.9, 101.5),  # partial hit, SL -> breakeven (100.0)
        (101.5, 101.6, 99.5, 99.8),  # dips back through breakeven
    ]
    df = build_candles(bars, freq="15min")
    signal = _signal(entry=100.0, sl=99.0, tp=103.0, generated_at=df["timestamp"].iloc[0])

    trade = simulate_trade(signal, df, signal_bar_idx=0, h1_analysis=_empty_h1_analysis(), symbol_cost=_ZERO_COST, dominant_confluence_family="liquidity")

    assert trade is not None
    assert trade.exit_reason == "breakeven_stop"
    assert trade.partial_taken is True
    # net_r = 0.5*1.5 + 0.5*0.0 = 0.75
    assert trade.gross_r == 0.75


def test_htf_choch_closes_full_before_partial():
    bars = [
        (100.0, 100.1, 99.9, 100.0),
        (100.0, 100.2, 99.9, 100.1),  # fill bar
        (100.1, 100.3, 100.0, 100.2),  # small favorable move, no partial yet
    ]
    df = build_candles(bars, freq="15min")
    signal = _signal(entry=100.0, sl=99.0, tp=103.0, generated_at=df["timestamp"].iloc[0])
    entry_time = df["timestamp"].iloc[1]
    choch_ts = df["timestamp"].iloc[2] + pd.Timedelta(minutes=15)  # within the 2nd managed bar's close
    h1_analysis = _h1_analysis_with_choch(choch_ts, "bearish")

    trade = simulate_trade(signal, df, signal_bar_idx=0, h1_analysis=h1_analysis, symbol_cost=_ZERO_COST, dominant_confluence_family="structure")

    assert trade is not None
    assert trade.exit_reason == "htf_invalidation_pre_partial"
    assert trade.partial_taken is False
    # exit at bar 2's close price (100.2): r = (100.2-100)/(100-99) = 0.2
    assert round(trade.gross_r, 4) == 0.2
