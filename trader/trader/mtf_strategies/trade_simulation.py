"""NUEVO -- necesario para la corrección del Prompt 5: saber CUÁNDO se cierra
una posición (por TP1+breakeven+TP2, por SL, o por salida temporal) es lo
único que permite implementar "un símbolo vuelve a estar disponible cuando
su posición se cierra" -- antes de esto, `mtf_strategies/` solo sabía
DETECTAR un setup, nunca simulaba su desenlace.

Mismo principio causal y misma convención conservadora que
`trader/backtest/trade.py` (que resuelve un tipo de señal distinto, del
pipeline de producción -- no se importa de ahí, se reimplementa acá para
mantener el aislamiento, pero la convención es la misma, ya auditada):
- Entrada: la PRIMERA vela M15 en o después de `fvg_confirmed_at` cuyo
  rango toca el nivel de entrada (50% de la FVG) -- fill al precio del
  nivel (es una entrada tipo límite a un precio específico, no una entrada
  a mercado en la apertura de la vela siguiente).
- Si en la misma vela se tocan SL y TP1 (o SL y TP2), se asume que el SL
  se tocó primero -- convención conservadora estándar cuando solo se tiene
  OHLC, no datos de tick.
- Salida temporal: SOLO antes del parcial (una vez en breakeven, el riesgo
  ya está protegido -- la regla 14 no dice nada sobre aplicar la salida
  temporal después del parcial, y aplicarla ahí inventaría una regla que la
  spec no tiene).

Esto NO calcula R-múltiplo ni resultado -- únicamente el timestamp de
cierre, para el propósito de concurrencia del Prompt 5. Calcular expectancy
o cualquier métrica de desempeño sigue fuera de alcance (ver Paso 3).
"""

from __future__ import annotations

import pandas as pd

from trader.events import TF_DURATION
from trader.mtf_strategies.exits import MAX_CANDLES_M15, MIN_PROGRESS_R, candles_elapsed_m15
from trader.mtf_strategies.signal import MTFSignal

_STEP = TF_DURATION["M15"]
_MAX_SEARCH_BARS = 700  # ~7 dias de M15 -- limite computacional pragmatico, no una regla de la spec


def progress_r(direction: str, entry: float, sl: float, bar_high: float, bar_low: float) -> float:
    """Progreso favorable (en R) alcanzado dentro de una vela -- pública
    porque `live_state.py` la reusa para la misma salida temporal pre-parcial
    en ejecución en vivo (un único cálculo de R, no dos implementaciones)."""
    risk = (entry - sl) if direction == "long" else (sl - entry)
    if risk <= 0:
        return 0.0
    favorable_extreme = bar_high if direction == "long" else bar_low
    moved = (favorable_extreme - entry) if direction == "long" else (entry - favorable_extreme)
    return moved / risk


def _simulate_close_from_entry(
    signal: MTFSignal, m15_df: pd.DataFrame, close_ts: pd.Series, entry_pos: int, entry_time: pd.Timestamp,
    temporal_exit: bool = True,
) -> tuple[pd.Timestamp | None, float | None]:
    """Cuerpo compartido por `simulate_position_close` (espera indefinida al
    retroceso, modelo original) y `simulate_pending_order` (orden límite real
    con expiración, migración 2026-09-23) -- una vez que se sabe CUÁNDO entró
    la posición, cómo se resuelve de ahí en más (parcial/breakeven/SL/TP2/
    salida temporal) es idéntico en ambos modelos.

    Devuelve `(cierre, R bruto)` -- R agregado 2026-09-25 para simular el
    corte diario -1.5R. Misma gestión que el piloto: 50% en TP1 + SL a
    breakeven, resto a TP2/BE; R contra el SL estructural original (igual que
    `live_state.realized_r_from_deals`). Sin costo -- lo resta el llamador."""
    is_long = signal.direction == "long"
    risk = abs(signal.entry - signal.sl)

    def r_at(price: float) -> float:
        return ((price - signal.entry) if is_long else (signal.entry - price)) / risk

    is_long = signal.direction == "long"
    partial_taken = False
    current_sl = signal.sl
    last_checked_pos = min(entry_pos + _MAX_SEARCH_BARS, len(m15_df) - 1)

    for pos in range(entry_pos, last_checked_pos + 1):
        row = m15_df.iloc[pos]
        bar_close = close_ts.iloc[pos]
        if bar_close <= entry_time:
            continue  # la propia vela de entrada no cuenta para chequear salidas -- recien la siguiente

        if temporal_exit and not partial_taken:  # False solo para medir la regla (2026-09-25), nunca en vivo
            progress = progress_r(signal.direction, signal.entry, signal.sl, row["high"], row["low"])
            bars_elapsed = candles_elapsed_m15(entry_time, bar_close)
            if bars_elapsed >= MAX_CANDLES_M15 and progress < MIN_PROGRESS_R:
                return bar_close, r_at(row["close"])  # salida temporal -- regla 14, antes del parcial únicamente

        sl_hit = (row["low"] <= current_sl) if is_long else (row["high"] >= current_sl)
        if sl_hit:
            # SL (o breakeven post-parcial) -- chequeado antes que TP, convención conservadora
            return bar_close, (0.5 * r_at(signal.tp1) if partial_taken else -1.0)

        if not partial_taken:
            tp1_hit = (row["high"] >= signal.tp1) if is_long else (row["low"] <= signal.tp1)
            if tp1_hit:
                partial_taken = True
                current_sl = signal.entry  # breakeven, regla 14
                continue
        else:
            tp2_hit = (row["high"] >= signal.tp2) if is_long else (row["low"] <= signal.tp2)
            if tp2_hit:
                return bar_close, 0.5 * r_at(signal.tp1) + 0.5 * r_at(signal.tp2)

    return None, None  # se acabaron los datos (o el límite de búsqueda) sin resolver


def simulate_position_close(signal: MTFSignal, m15_df: pd.DataFrame) -> pd.Timestamp | None:
    """Devuelve el timestamp de cierre de la posición, o None si nunca se
    resuelve dentro de `m15_df` (los datos se acaban antes, o pasa
    `_MAX_SEARCH_BARS` sin resolución -- ambos casos tratados igual, el
    símbolo queda "abierto" hasta el final de la ventana disponible).
    Modelo ORIGINAL (mercado, espera indefinida al retroceso) -- para el
    modelo de orden límite real con expiración, ver `simulate_pending_order`."""
    ts = m15_df["timestamp"]
    close_ts = ts + _STEP
    is_long = signal.direction == "long"

    after_fvg = m15_df[close_ts >= signal.fvg_confirmed_at]
    if len(after_fvg) == 0:
        return None
    touch_mask = (after_fvg["low"] <= signal.entry) if is_long else (after_fvg["high"] >= signal.entry)
    touched = after_fvg[touch_mask]
    if len(touched) == 0:
        return None
    entry_pos = m15_df.index.get_loc(touched.index[0])
    entry_time = close_ts.iloc[entry_pos]
    return _simulate_close_from_entry(signal, m15_df, close_ts, entry_pos, entry_time)[0]


def simulate_pending_order(signal: MTFSignal, m15_df: pd.DataFrame, deadline: pd.Timestamp) -> tuple[bool, pd.Timestamp | None]:
    filled, freed_at, _, _ = simulate_pending_order_outcome(signal, m15_df, deadline)
    return filled, freed_at


def simulate_pending_order_outcome(
    signal: MTFSignal, m15_df: pd.DataFrame, deadline: pd.Timestamp, temporal_exit: bool = True,
) -> tuple[bool, pd.Timestamp | None, pd.Timestamp | None, float | None]:
    """Migración a orden límite real (2026-09-23, ver
    project_mtf_pending_limit_orders_2026-09-23): la entrada ya NO espera
    indefinidamente el retroceso -- una orden límite real expira si el precio
    no toca el nivel antes de `deadline`. El cálculo de `deadline` es
    responsabilidad del llamador (offset fijo de 90 min en la primera versión
    de este modelo; fin de sesión de entrada -- `session_risk.entry_session_end`
    -- desde el Prompt "ventana de expiración = fin de sesión", 2026-09-23,
    comparación única contra el offset fijo) -- esta función solo sabe
    "¿tocó el nivel antes de esta fecha?", no de dónde sale la fecha.

    Devuelve `(se_llenó, cuándo_se_libera_el_símbolo)`: si nunca se llena
    dentro de la ventana, el símbolo se libera al vencer la orden (nunca
    comprometió capital, no hay posición que resolver); si se llena, se
    libera cuando la posición resultante cierra (mismo motor de simulación
    que `simulate_position_close` de ahí en más -- parcial/breakeven/SL/TP2/
    salida temporal, sin duplicar esa lógica)."""
    ts = m15_df["timestamp"]
    close_ts = ts + _STEP
    is_long = signal.direction == "long"

    window = m15_df[(close_ts >= signal.fvg_confirmed_at) & (close_ts <= deadline)]
    touch_mask = (window["low"] <= signal.entry) if is_long else (window["high"] >= signal.entry)
    touched = window[touch_mask]
    if len(touched) == 0:
        return False, deadline, None, None

    entry_pos = m15_df.index.get_loc(touched.index[0])
    entry_time = close_ts.iloc[entry_pos]
    closed_at, r = _simulate_close_from_entry(signal, m15_df, close_ts, entry_pos, entry_time, temporal_exit)
    return True, closed_at, entry_time, r
