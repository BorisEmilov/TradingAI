import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from tradingai.ai.evaluation.backtester import Backtester, Trade, summarize
from tradingai.core.signal import Direction, TradingSignal


def _candles(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="15min"),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        }
    )


def _long_signal(entry: float, sl: float, tp: float) -> TradingSignal:
    return TradingSignal(
        symbol="EURUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=Direction.LONG,
        confidence=0.8,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


def test_costs_reduce_pnl_on_winning_trade():
    # Precio sube directo hasta el TP.
    candles = _candles([1.1000, 1.1010, 1.1020, 1.1050, 1.1100])
    signal = _long_signal(entry=1.1000, sl=1.0950, tp=1.1100)

    no_costs = Backtester(confidence_threshold=0.5, spread_pips=0, slippage_pips=0, commission_pips=0)
    with_costs = Backtester(confidence_threshold=0.5, spread_pips=1.0, slippage_pips=0.2, commission_pips=0.3)

    trade_no_costs = no_costs.run(candles, [(0, signal)])[0]
    trade_with_costs = with_costs.run(candles, [(0, signal)])[0]

    assert trade_with_costs.pnl_pct < trade_no_costs.pnl_pct
    # 1.5 pips de coste total en un TP de 100 pips: la diferencia deberia ser pequeña pero
    # medible, y ambas deberian seguir siendo ganadoras.
    assert trade_with_costs.pnl_pct > 0
    assert trade_no_costs.pnl_pct > 0


def test_costs_can_flip_a_marginal_win_into_a_loss():
    # El precio nunca toca TP/SL y se cierra por timeout con una ganancia minima —
    # mas chica que el coste total, deberia volverse perdida al restar costes.
    candles = _candles([1.1000] + [1.10005] * 10)
    signal = _long_signal(entry=1.1000, sl=1.0950, tp=1.2000)

    no_costs = Backtester(confidence_threshold=0.5, max_holding_bars=10, spread_pips=0, slippage_pips=0)
    with_costs = Backtester(confidence_threshold=0.5, max_holding_bars=10, spread_pips=1.0, slippage_pips=0.5)

    trade_no_costs = no_costs.run(candles, [(0, signal)])[0]
    trade_with_costs = with_costs.run(candles, [(0, signal)])[0]

    assert trade_no_costs.pnl_pct > 0
    assert trade_with_costs.pnl_pct < 0


def _trade_with_pnl(pnl_pct: float) -> Trade:
    return Trade(signal=_long_signal(1.1000, 1.0950, 1.1100), exit_price=1.1000, exit_reason="tp", pnl_pct=pnl_pct)


def test_summarize_computes_risk_adjusted_metrics():
    # Calculado a mano: mean=0.006, std=0.018547, sharpe=mean/std, downside_std=0.01
    # (solo -0.01 y -0.02 contribuyen), sortino=0.6, drawdown maximo=0.02 (tras el
    # pico acumulado de 0.04 en la posicion 3, cae hasta 0.02 en la posicion 4).
    pnls = [0.02, -0.01, 0.03, -0.02, 0.01]
    trades = [_trade_with_pnl(p) for p in pnls]

    stats = summarize(trades)

    assert stats["n_trades"] == 5
    assert stats["win_rate"] == pytest.approx(3 / 5)
    assert stats["avg_pnl_pct"] == pytest.approx(0.006)
    assert stats["sharpe"] == pytest.approx(0.3234983196103152)
    assert stats["sortino"] == pytest.approx(0.6)
    assert stats["max_drawdown_pct"] == pytest.approx(0.02)


def test_summarize_handles_no_losses_without_division_by_zero():
    trades = [_trade_with_pnl(p) for p in [0.01, 0.02, 0.01]]
    stats = summarize(trades)
    assert stats["sortino"] == 0.0  # sin perdidas -> downside_std=0, se evita division por cero
    assert stats["max_drawdown_pct"] == 0.0  # nunca cae por debajo del pico


def test_summarize_empty_trades():
    assert summarize([]) == {"n_trades": 0}
