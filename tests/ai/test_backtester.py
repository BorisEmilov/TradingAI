from datetime import datetime, timezone

import pandas as pd

from tradingai.ai.evaluation.backtester import Backtester
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
