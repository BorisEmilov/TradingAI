"""Gestion de salida basada en estructura, mas alla de un TP/SL fijo (2026-08-27).

Dos mecanismos, ambos reutilizando la MISMA feature de sesgo (`bias_bullish`/
`bias_bearish`, alineacion de EMA20/50/200) que ya ve el modelo de entrada -- ver
`ai.data.features.indicators.add_indicators()`. No es un componente de decision
nuevo: es la misma señal de estructura que ya se entrena y valida, aplicada tambien
a operaciones ya abiertas en vez de solo a la entrada.

1. `structure_invalidated`: si el sesgo del simbolo se voltea en contra de la
   direccion de la operacion, la estructura que justifico la entrada ya no existe --
   permite cerrar antes de esperar a que el precio llegue al SL fijo.
2. `compute_dynamic_take_profit`: si el sesgo se mantiene a favor y aparece un swing
   de estructura mas alla del TP actual, extiende el TP (ratchet, nunca lo acerca) en
   vez de cerrar en un multiplo fijo de riesgo -- deja correr al ganador mientras la
   estructura lo respalde, permitiendo 1:3/1:4 sin renunciar al piso de 1:2 (el TP
   original nunca se acerca, solo se aleja).
"""

from __future__ import annotations

import pandas as pd

from tradingai.ai.data.features.indicators import add_indicators
from tradingai.core.signal import Direction
from tradingai.core.structure import last_confirmed_swing_high, last_confirmed_swing_low


def structure_invalidated(candles: pd.DataFrame, direction: Direction) -> bool:
    """True si el sesgo de EMAs ya se volteo EN CONTRA de la direccion de la
    operacion en la ultima vela cerrada.

    Solo un sesgo contrario claro (bullish/bearish) invalida -- `bias_neutral`
    (EMAs no alineadas con claridad, ej. rango) NO cierra la operacion; exigir un
    sesgo opuesto explicito evita cerrar por ruido en cuanto el precio se aplana un
    momento.
    """
    with_bias = add_indicators(candles, include=["ema"])
    last = with_bias.iloc[-1]
    if direction == Direction.LONG:
        return bool(last["bias_bearish"])
    if direction == Direction.SHORT:
        return bool(last["bias_bullish"])
    return False


def compute_dynamic_take_profit(
    candles: pd.DataFrame,
    direction: Direction,
    current_tp: float,
    current_price: float,
    swing_left: int = 3,
    swing_right: int = 3,
) -> float | None:
    """Devuelve un TP nuevo mas lejano si aparecio un swing de estructura mas alla
    del TP actual a favor del precio, o None si no hay que tocarlo.

    Ratchet igual que el trailing stop (`mt5.trailing_stop`): solo EXTIENDE el TP
    (nunca lo acerca), y solo si el nuevo nivel todavia deja margen real por delante
    del precio actual (si no, se ignora en vez de fijar un TP ya alcanzado).
    """
    if direction == Direction.LONG:
        swing = last_confirmed_swing_high(candles, swing_left, swing_right)
        if swing is None or swing <= current_tp or swing <= current_price:
            return None
        return swing

    if direction == Direction.SHORT:
        swing = last_confirmed_swing_low(candles, swing_left, swing_right)
        if swing is None or swing >= current_tp or swing >= current_price:
            return None
        return swing

    return None
