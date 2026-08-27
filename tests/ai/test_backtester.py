import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from tradingai.ai.evaluation.backtester import (
    Backtester,
    Trade,
    default_spread_pips,
    pip_size,
    resolve_spread_pips,
    summarize,
)
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


def _trade_with_pnl(pnl_pct: float, entry: float = 1.1000, sl: float = 1.0950) -> Trade:
    return Trade(signal=_long_signal(entry, sl, 1.1100), exit_price=entry, exit_reason="tp", pnl_pct=pnl_pct)


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
    # risk_pct = (1.1000-1.0950)/1.1000 = 0.00454545... para las 5 (mismo signal) ->
    # expectancy_r = avg_pnl_pct / risk_pct = 0.006 / 0.00454545... = 1.32
    assert stats["expectancy_r"] == pytest.approx(1.32)
    # gross_profit=0.06 (0.02+0.03+0.01), gross_loss=0.03 (0.01+0.02) -> 2.0
    assert stats["profit_factor"] == pytest.approx(2.0)


def test_summarize_handles_no_losses_without_division_by_zero():
    trades = [_trade_with_pnl(p) for p in [0.01, 0.02, 0.01]]
    stats = summarize(trades)
    assert stats["sortino"] == 0.0  # sin perdidas -> downside_std=0, se evita division por cero
    assert stats["max_drawdown_pct"] == 0.0  # nunca cae por debajo del pico
    assert stats["profit_factor"] == float("inf")  # sin perdidas -> profit factor "perfecto"


def test_summarize_profit_factor_zero_when_no_wins_and_no_losses():
    stats = summarize([_trade_with_pnl(0.0)])
    assert stats["profit_factor"] == 0.0


def test_summarize_expectancy_r_normalizes_by_each_trade_own_risk():
    # Misma pnl_pct (0.01) pero con distinta distancia de SL -> distinto R arriesgado:
    # trade A arriesga 0.05 (risk_pct~0.0455), trade B arriesga 0.10 (risk_pct~0.0909).
    # El pnl_pct crudo promedio seria igual (0.01) pero en R son muy distintos.
    trade_a = _trade_with_pnl(0.01, entry=1.1000, sl=1.0950)
    trade_b = _trade_with_pnl(0.01, entry=1.1000, sl=1.0900)

    stats = summarize([trade_a, trade_b])

    risk_a = (1.1000 - 1.0950) / 1.1000
    risk_b = (1.1000 - 1.0900) / 1.1000
    expected = ((0.01 / risk_a) + (0.01 / risk_b)) / 2
    assert stats["expectancy_r"] == pytest.approx(expected)
    assert stats["avg_pnl_pct"] == pytest.approx(0.01)  # el crudo no distingue el riesgo


def test_summarize_empty_trades():
    assert summarize([]) == {"n_trades": 0}


def test_pip_size_forex_default():
    assert pip_size("EURUSD") == 0.0001


def test_pip_size_jpy_pairs():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("GBPJPY") == 0.01


def test_pip_size_index_uses_point_not_forex_pip():
    # Bug real del 2026-08-26: US500 cotiza con point=0.01 en MT5, no 0.00001 como
    # un par forex -- usar el pip de forex ahi era un error de 2 ordenes de magnitud.
    assert pip_size("US500") == 0.01


def test_default_spread_pips_majors_vs_crosses():
    assert default_spread_pips("EURUSD") == 1.0
    assert default_spread_pips("USDCAD") == 1.0
    assert default_spread_pips("EURAUD") == 2.0
    assert default_spread_pips("GBPCAD") == 2.0


def test_resolve_spread_pips_prefers_measured_value_over_default():
    cfg = {"spread_pips_by_symbol": {"EURAUD": 3.2}}
    assert resolve_spread_pips(cfg, "EURAUD") == 3.2
    assert resolve_spread_pips(cfg, "EURUSD") == default_spread_pips("EURUSD")


def test_resolve_spread_pips_falls_back_to_default_when_map_empty():
    assert resolve_spread_pips({}, "GBPAUD") == default_spread_pips("GBPAUD")
    assert resolve_spread_pips({"spread_pips_by_symbol": {}}, "GBPAUD") == default_spread_pips("GBPAUD")
