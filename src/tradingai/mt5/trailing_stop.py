"""Trailing stop-loss basado en estructura de swings.

Una vez que una operacion alcanza cierto multiplo de R en beneficio, mueve el SL a la
ultima zona de invalidacion estructural confirmada a favor del precio (el ultimo swing
low para un LONG, el ultimo swing high para un SHORT) en vez de una distancia fija o
un trailing por ATR "a ciegas" -- ver conversacion del 2026-08-25: el usuario pidio
explicitamente que el SL se mueva "con sentido, en zonas clave" y no de forma
aleatoria.

Un swing (fractal) se confirma cuando una vela es mas extrema que `swing_left` velas
anteriores Y `swing_right` velas posteriores -- mismo criterio de confirmacion que ya
se usa para los pivotes de `ai.data.features.elliott` (no se reactiva ante el ultimo
minimo/maximo, que todavia puede romperse).

El movimiento es SIEMPRE un ratchet: el nuevo SL solo se acepta si protege mas
beneficio que el actual (nunca se afloja), y solo si deja margen real hasta el precio
actual (si no, se ignora en vez de arriesgar un cierre inmediato).
"""

from __future__ import annotations

import pandas as pd

from tradingai.core.signal import Direction
from tradingai.core.structure import last_confirmed_swing_high, last_confirmed_swing_low


def compute_trailing_sl(
    candles: pd.DataFrame,
    direction: Direction,
    entry_price: float,
    initial_risk: float,
    current_sl: float,
    current_price: float,
    r_multiple_to_activate: float = 1.0,
    swing_left: int = 3,
    swing_right: int = 3,
) -> float | None:
    """Devuelve el nuevo SL si corresponde moverlo, o None si no hay que tocarlo.

    No se activa hasta que el precio se movio a favor al menos
    `r_multiple_to_activate` veces `initial_risk` (la distancia de riesgo ORIGINAL,
    entry - SL en el momento de abrir -- pasada explicitamente por el llamador, NO
    recalculada de `current_sl` en cada llamada). Esto importa porque `current_sl`
    puede haber cambiado desde entonces (trailing previo, o un cierre parcial que
    movio el SL a breakeven -- ver mt5.scaled_exit): si se recalculara
    `entry_price - current_sl` aca, un SL en breakeven daria risk=0 y el trailing se
    romperia en silencio para siempre en esa operacion (bug real encontrado el
    2026-08-26 al combinar esto con el cierre parcial).

    A partir de ahi, busca el ultimo swing confirmado a favor del precio y solo lo
    acepta como nuevo SL si mejora a `current_sl` y sigue dejando margen hasta el
    precio de mercado.
    """
    if initial_risk <= 0:
        return None

    if direction == Direction.LONG:
        profit_distance = current_price - entry_price
        if profit_distance < r_multiple_to_activate * initial_risk:
            return None
        swing = last_confirmed_swing_low(candles, swing_left, swing_right)
        if swing is None or swing <= current_sl or swing >= current_price:
            return None
        return swing

    if direction == Direction.SHORT:
        profit_distance = entry_price - current_price
        if profit_distance < r_multiple_to_activate * initial_risk:
            return None
        swing = last_confirmed_swing_high(candles, swing_left, swing_right)
        if swing is None or swing >= current_sl or swing <= current_price:
            return None
        return swing

    return None
