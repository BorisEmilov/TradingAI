"""Backtest simplificado: simula la ejecucion de senales del modelo sobre historico.

Modela spread, slippage y comision como un coste fijo en pips por operacion (se paga
siempre, gane o pierda — asi es como funciona en la realidad). No modela ejecucion
multi-posicion ni variacion del spread segun volatilidad/horario; sirve para validar
si el modelo tiene edge real una vez descontados los costes basicos, antes de pasar a
un backtest mas riguroso o a paper trading en MT5.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingai.core.signal import Direction, TradingSignal


@dataclass
class Trade:
    signal: TradingSignal
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "timeout"
    pnl_pct: float


class Backtester:
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_holding_bars: int = 50,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.2,
        commission_pips: float = 0.0,
        pip_size: float = 0.0001,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_holding_bars = max_holding_bars
        # Coste de ida y vuelta (spread + slippage + comision-equivalente) en precio,
        # restado del resultado de cada operacion sin importar si gana o pierde.
        # commission_pips es el equivalente en pips de la comision del broker (0 en
        # cuentas solo-spread; las cuentas ECN suelen cobrar comision aparte).
        self.cost_price = (spread_pips + slippage_pips + commission_pips) * pip_size

    def run(self, candles: pd.DataFrame, signals: list[tuple[int, TradingSignal]]) -> list[Trade]:
        """`signals` es una lista de (indice_en_candles, TradingSignal)."""
        trades = []
        for idx, signal in signals:
            if not signal.is_actionable(self.confidence_threshold):
                continue
            if signal.entry_price is None or signal.take_profit is None or signal.stop_loss is None:
                continue

            trade = self._simulate_trade(candles, idx, signal)
            if trade:
                trades.append(trade)
        return trades

    def _simulate_trade(self, candles: pd.DataFrame, idx: int, signal: TradingSignal) -> Trade | None:
        future = candles.iloc[idx + 1 : idx + 1 + self.max_holding_bars]
        if future.empty:
            return None

        is_long = signal.direction == Direction.LONG
        for _, bar in future.iterrows():
            hit_tp = bar["high"] >= signal.take_profit if is_long else bar["low"] <= signal.take_profit
            hit_sl = bar["low"] <= signal.stop_loss if is_long else bar["high"] >= signal.stop_loss

            if hit_tp:
                return self._make_trade(signal, signal.take_profit, "tp", is_long)
            if hit_sl:
                return self._make_trade(signal, signal.stop_loss, "sl", is_long)

        last_close = future["close"].iloc[-1]
        return self._make_trade(signal, last_close, "timeout", is_long)

    def _make_trade(self, signal: TradingSignal, exit_price: float, reason: str, is_long: bool) -> Trade:
        price_diff = exit_price - signal.entry_price
        directional_diff = price_diff if is_long else -price_diff
        net_diff = directional_diff - self.cost_price
        pnl_pct = net_diff / signal.entry_price
        return Trade(signal, exit_price, reason, pnl_pct)


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n_trades": 0}

    wins = [t for t in trades if t.pnl_pct > 0]
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_pnl_pct": sum(t.pnl_pct for t in trades) / len(trades),
        "total_pnl_pct": sum(t.pnl_pct for t in trades),
    }
