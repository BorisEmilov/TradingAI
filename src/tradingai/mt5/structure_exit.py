"""Gestion de salida basada en estructura, mas alla de un TP/SL fijo (2026-08-27,
redefinido 2026-08-28).

Dos mecanismos, ambos basados en la MISMA deteccion de swings confirmados
(`core.structure`) que ya usa el trailing stop -- estructura de precio real (maximos
y minimos confirmados por velas), no un indicador derivado como una media movil.
La primera version (2026-08-27) usaba el apilamiento de EMA20/50/200 para decidir
"sigue a favor la estructura?"; se reemplazo el mismo dia por pedido explicito del
usuario ("tenemos que enfocarnos en estructura de velas y estructura general no
medias moviles") -- una media de 200 periodos es lenta y no es estructura de precio,
es un promedio.

1. `structure_invalidated`: ruptura de estructura (Break of Structure / Change of
   Character) -- si el ultimo swing confirmado en contra de la operacion es MAS
   extremo que el swing confirmado anterior (un minimo mas bajo estando en largo, o
   un maximo mas alto estando en corto), la secuencia de estructura que sostenia la
   operacion se rompio. Permite cerrar antes de esperar a que el precio llegue al SL
   fijo.
2. `compute_dynamic_take_profit`: si aparece un swing de estructura mas alla del TP
   actual, lo extiende (ratchet, nunca lo acerca) en vez de cerrar en un multiplo
   fijo de riesgo -- deja correr al ganador mientras la estructura lo respalde,
   permitiendo 1:3/1:4 sin renunciar al piso de 1:2 (el TP original nunca se acerca,
   solo se aleja). Sin cambios respecto a la version anterior.
"""

from __future__ import annotations

import pandas as pd

from tradingai.core.signal import Direction
from tradingai.core.structure import (
    confirmed_swing_highs,
    confirmed_swing_lows,
    last_confirmed_swing_high,
    last_confirmed_swing_low,
)


def structure_invalidated(
    candles: pd.DataFrame, direction: Direction, swing_left: int = 3, swing_right: int = 3
) -> bool:
    """True si la secuencia de swings se rompio EN CONTRA de la direccion de la
    operacion -- un minimo mas bajo que el minimo confirmado anterior estando en
    largo (o un maximo mas alto que el anterior estando en corto).

    Necesita al menos 2 swings confirmados del lado relevante para poder comparar --
    si todavia no hay suficiente estructura formada, no invalida (no hay evidencia
    de ruptura, no asumir la peor).
    """
    if direction == Direction.LONG:
        lows = confirmed_swing_lows(candles, swing_left, swing_right, count=2)
        if len(lows) < 2:
            return False
        return lows[-1] < lows[-2]

    if direction == Direction.SHORT:
        highs = confirmed_swing_highs(candles, swing_left, swing_right, count=2)
        if len(highs) < 2:
            return False
        return highs[-1] > highs[-2]

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
