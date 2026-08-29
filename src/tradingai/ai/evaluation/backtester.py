"""Backtest simplificado: simula la ejecucion de senales del modelo sobre historico.

Modela spread, slippage y comision como un coste fijo en pips por operacion (se paga
siempre, gane o pierda — asi es como funciona en la realidad). No modela ejecucion
multi-posicion ni variacion del spread segun volatilidad/horario; sirve para validar
si el modelo tiene edge real una vez descontados los costes basicos, antes de pasar a
un backtest mas riguroso o a paper trading en MT5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from tradingai.core.instruments import pip_size  # noqa: F401 -- reexportado, ver abajo
from tradingai.core.signal import Direction, TradingSignal
from tradingai.core.structure import (
    confirmed_swing_highs,
    confirmed_swing_lows,
    last_confirmed_swing_high,
    last_confirmed_swing_low,
)

# Simbolos "major" (USD contra la otra divisa mas liquida de su categoria): en la
# practica retail suelen tener el spread mas bajo/estable. Todo lo demas en la lista
# de simbolos del piloto son pares cruzados (sin USD) -- tipicamente mas caros por
# menor liquidez.
_MAJOR_SYMBOLS = {"EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF"}

# Sin datos reales de spread por broker todavia (ver conversacion 2026-08-26): en vez
# de un valor fijo unico para los 15 simbolos (subestima los cruces, que en retail
# suelen costar mas que los majors por menor liquidez), se aplica un ajuste
# CONSERVADOR por categoria. Actualizar `spread_pips_by_symbol` en config.yaml en
# cuanto haya cifras reales del broker -- estos son solo un piso razonable, no una
# medicion.
_MAJOR_SPREAD_PIPS_DEFAULT = 1.0
_CROSS_SPREAD_PIPS_DEFAULT = 2.0

# `pip_size` vive ahora en `tradingai.core.instruments` (2026-08-27): tanto este
# modulo (evaluacion/backtest) como `ai.inference.gbm_predictor` (piso minimo de SL
# en vivo) lo necesitan, y ninguno de los dos debia depender del otro. Se reexporta
# aca para no romper los scripts existentes (backtest.py/baseline_gbm.py/
# walk_forward.py) que ya hacen `from tradingai.ai.evaluation.backtester import pip_size`.


def default_spread_pips(symbol: str) -> float:
    """Spread por defecto si no hay un valor real medido para este simbolo (ver
    `spread_pips_by_symbol` en config.yaml)."""
    return _MAJOR_SPREAD_PIPS_DEFAULT if symbol.upper() in _MAJOR_SYMBOLS else _CROSS_SPREAD_PIPS_DEFAULT


def resolve_spread_pips(backtest_cfg: dict, symbol: str) -> float:
    """Spread a usar para `symbol`: el medido en `spread_pips_by_symbol` (config.yaml)
    si existe, si no el default conservador por categoria (`default_spread_pips`)."""
    by_symbol = backtest_cfg.get("spread_pips_by_symbol") or {}
    return by_symbol.get(symbol, default_spread_pips(symbol))


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
        dynamic_exit: bool = False,
        structure_swing_left: int = 3,
        structure_swing_right: int = 3,
        structure_lookback_candles: int = 100,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_holding_bars = max_holding_bars
        # Coste de ida y vuelta (spread + slippage + comision-equivalente) en precio,
        # restado del resultado de cada operacion sin importar si gana o pierde.
        # commission_pips es el equivalente en pips de la comision del broker (0 en
        # cuentas solo-spread; las cuentas ECN suelen cobrar comision aparte).
        self.cost_price = (spread_pips + slippage_pips + commission_pips) * pip_size
        # Modo de validacion para mt5.structure_exit: en vez de TP/SL fijos, cierra
        # antes si la secuencia de swings se rompe en contra (Break of
        # Structure/Change of Character, ver core.structure -- redefinido el
        # 2026-08-28, antes usaba el apilamiento de EMA20/50/200 por pedido explicito
        # del usuario de usar estructura de velas real en vez de una media movil), y
        # extiende el TP hacia el siguiente swing mientras la estructura siga a
        # favor. El SL se deja FIJO en ambos modos (la gestion de SL --
        # breakeven/trailing -- ya esta en vivo y se evalua por separado via los CSV
        # de tracking, no aca) para aislar el efecto de ESTA propuesta especifica.
        self.dynamic_exit = dynamic_exit
        self.structure_swing_left = structure_swing_left
        self.structure_swing_right = structure_swing_right
        self.structure_lookback_candles = structure_lookback_candles

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
        if self.dynamic_exit:
            return self._simulate_trade_dynamic(candles, idx, future, signal, is_long)

        for _, bar in future.iterrows():
            hit_tp = bar["high"] >= signal.take_profit if is_long else bar["low"] <= signal.take_profit
            hit_sl = bar["low"] <= signal.stop_loss if is_long else bar["high"] >= signal.stop_loss

            if hit_tp:
                return self._make_trade(signal, signal.take_profit, "tp", is_long)
            if hit_sl:
                return self._make_trade(signal, signal.stop_loss, "sl", is_long)

        last_close = future["close"].iloc[-1]
        return self._make_trade(signal, last_close, "timeout", is_long)

    def _simulate_trade_dynamic(
        self, candles: pd.DataFrame, idx: int, future: pd.DataFrame, signal: TradingSignal, is_long: bool
    ) -> Trade | None:
        """Replica `mt5.structure_exit` sobre historico: cierra antes si la
        secuencia de swings se rompe en contra (BOS/CHoCH), o extiende el TP hacia
        el siguiente swing mientras la estructura siga a favor. El SL se evalua
        siempre contra el nivel FIJO original (ver `dynamic_exit` en `__init__`)."""
        current_tp = signal.take_profit

        for offset, (_, bar) in enumerate(future.iterrows()):
            abs_idx = idx + 1 + offset
            swing_window = candles.iloc[max(0, abs_idx - self.structure_lookback_candles + 1) : abs_idx + 1]

            if is_long:
                recent_lows = confirmed_swing_lows(swing_window, self.structure_swing_left, self.structure_swing_right, count=2)
                invalidated = len(recent_lows) >= 2 and recent_lows[-1] < recent_lows[-2]
            else:
                recent_highs = confirmed_swing_highs(swing_window, self.structure_swing_left, self.structure_swing_right, count=2)
                invalidated = len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]
            if invalidated:
                return self._make_trade(signal, bar["close"], "estructura", is_long)
            if is_long:
                swing = last_confirmed_swing_high(swing_window, self.structure_swing_left, self.structure_swing_right)
                if swing is not None and swing > current_tp and swing > bar["close"]:
                    current_tp = swing
            else:
                swing = last_confirmed_swing_low(swing_window, self.structure_swing_left, self.structure_swing_right)
                if swing is not None and swing < current_tp and swing < bar["close"]:
                    current_tp = swing

            hit_tp = bar["high"] >= current_tp if is_long else bar["low"] <= current_tp
            hit_sl = bar["low"] <= signal.stop_loss if is_long else bar["high"] >= signal.stop_loss

            if hit_tp:
                return self._make_trade(signal, current_tp, "tp", is_long)
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
    """Win rate/pnl (ya existian) + metricas ajustadas a riesgo, no solo el retorno
    bruto: un 55% de acierto con drawdowns brutales no es lo mismo que uno suave.

    Sharpe/Sortino aqui son "por operacion" (no anualizados: anualizar exigiria
    asumir una frecuencia de trading fija, que varia por simbolo/regimen) -- sirven
    para comparar la CALIDAD del retorno entre folds/simbolos con la misma unidad,
    no como un Sharpe de cartera clasico.

    `expectancy_r` y `profit_factor` (2026-08-26): el % de aciertos aislado no dice
    si el sistema es rentable -- un 40% de acierto puede ser muy bueno si se gana 2.5x
    lo que se pierde. `expectancy_r` es el promedio de cuantas veces el riesgo
    arriesgado se gana/pierde por operacion (pnl_pct / distancia_SL_pct de ESA
    operacion, promediado) -- a diferencia de `avg_pnl_pct`, es comparable entre
    simbolos/configs con distinta distancia de SL/ATR, que es justo lo que varia
    entre los 15 simbolos del piloto. `profit_factor` = ganancia bruta / perdida
    bruta; con 0 operaciones perdedoras da `inf` (caso real de muestras chicas: no
    tratar un profit_factor altisimo con pocas operaciones como señal fuerte, ver
    `n_trades` siempre junto a el).
    """
    if not trades:
        return {"n_trades": 0}

    pnls = [t.pnl_pct for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mean_pnl = sum(pnls) / n

    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    r_multiples = []
    for t in trades:
        risk_pct = abs(t.signal.entry_price - t.signal.stop_loss) / t.signal.entry_price
        if risk_pct > 0:
            r_multiples.append(t.pnl_pct / risk_pct)
    expectancy_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

    variance = sum((p - mean_pnl) ** 2 for p in pnls) / n
    std_pnl = math.sqrt(variance)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    # Desviacion a la baja (semi-desviacion respecto a 0): solo penaliza la
    # volatilidad de las perdidas, no la de las ganancias -- por eso Sortino no
    # castiga a un sistema por tener operaciones muy ganadoras ocasionales.
    downside_variance = sum(p**2 for p in pnls if p < 0) / n
    downside_std = math.sqrt(downside_variance)
    sortino = mean_pnl / downside_std if downside_std > 0 else 0.0

    running = 0.0
    peak = float("-inf")
    max_drawdown = 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        max_drawdown = max(max_drawdown, peak - running)

    return {
        "n_trades": n,
        "win_rate": len(wins) / n,
        "avg_pnl_pct": mean_pnl,
        "total_pnl_pct": sum(pnls),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown,
        "expectancy_r": expectancy_r,
        "profit_factor": profit_factor,
    }
