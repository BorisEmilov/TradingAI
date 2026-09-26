import pandas as pd

from trader.backtest.metrics import breakdown_by_category, breakdown_by_session, breakdown_by_symbol, compute_metrics
from trader.backtest.trade import TradeResult

_BASE_TS = pd.Timestamp("2026-01-05", tz="UTC")


def _trade(
    net_r: float, exit_offset_hours: int, symbol: str = "EURUSD", session_labels=None, category: str = "trend"
) -> TradeResult:
    return TradeResult(
        symbol=symbol,
        direction="long",
        category=category,
        signal_time=_BASE_TS,
        entry_time=_BASE_TS,
        entry=100.0,
        sl=99.0,
        tp=103.0,
        planned_rr=3.0,
        actual_rr=3.0,
        partial_taken=False,
        exit_time=_BASE_TS + pd.Timedelta(hours=exit_offset_hours),
        exit_bar_index=exit_offset_hours,
        exit_reason="take_profit" if net_r > 0 else "stop_loss",
        gross_r=net_r,
        cost_r=0.0,
        net_r=net_r,
        session_labels=session_labels or ["Londres"],
        confirmation_kind="choch",
        dominant_confluence_family="structure",
    )


def test_compute_metrics_matches_hand_calculation():
    trades = [_trade(r, i) for i, r in enumerate([1.5, -1.0, 2.0, -1.0, -1.0], start=1)]

    m = compute_metrics(trades)

    assert m.n_trades == 5
    assert m.win_rate == 2 / 5
    assert round(m.expectancy_r, 6) == round(0.5 / 5, 6)
    assert round(m.profit_factor, 6) == round(3.5 / 3.0, 6)
    assert round(m.max_drawdown_r, 6) == 2.0


def test_compute_metrics_empty_list():
    m = compute_metrics([])
    assert m.n_trades == 0
    assert m.expectancy_r == 0.0
    assert m.profit_factor == 0.0
    assert m.max_drawdown_r == 0.0


def test_profit_factor_infinite_when_no_losses():
    trades = [_trade(1.0, 1), _trade(2.0, 2)]
    m = compute_metrics(trades)
    assert m.profit_factor == float("inf")


def test_breakdown_by_symbol_groups_correctly():
    trades = [_trade(1.0, 1, symbol="EURUSD"), _trade(-1.0, 2, symbol="EURUSD"), _trade(2.0, 3, symbol="GBPUSD")]
    grouped = breakdown_by_symbol(trades)

    assert set(grouped.keys()) == {"EURUSD", "GBPUSD"}
    assert grouped["EURUSD"].n_trades == 2
    assert grouped["GBPUSD"].n_trades == 1
    assert grouped["GBPUSD"].expectancy_r == 2.0


def test_breakdown_by_session_joins_active_labels():
    trades = [
        _trade(1.0, 1, session_labels=["Londres", "NY"]),
        _trade(-1.0, 2, session_labels=["Asia"]),
    ]
    grouped = breakdown_by_session(trades)
    assert set(grouped.keys()) == {"Londres/NY", "Asia"}


def test_breakdown_by_category_separates_trend_and_reversal():
    trades = [
        _trade(1.0, 1, category="trend"),
        _trade(-1.0, 2, category="trend"),
        _trade(2.0, 3, category="reversal"),
    ]
    grouped = breakdown_by_category(trades)

    assert set(grouped.keys()) == {"trend", "reversal"}
    assert grouped["trend"].n_trades == 2
    assert grouped["reversal"].n_trades == 1
    assert grouped["reversal"].expectancy_r == 2.0
