"""Metadata de instrumentos compartida entre evaluacion (backtest) e inferencia en
vivo (prediccion) -- vive en `core` porque ambos lados dependen de ella y ninguno
deberia depender del otro (ver bug real del 2026-08-26: `_pip_size` vivia
triplicada en 3 scripts antes de consolidarse; se corrigio a un solo lugar, pero el
lugar elegido entonces -- `ai.evaluation.backtester` -- no era el correcto para que
`ai.inference` tambien lo usara sin una dependencia cruzada rara).
"""

from __future__ import annotations

# Los indices (US500) cotizan en puntos, no en "pips" de forex -- un piso de SL
# pensado en pips de forex no tiene sentido dimensional ahi (ver `min_sl_distance_price`).
_INDEX_SYMBOLS = {"US500"}


def pip_size(symbol: str) -> float:
    """Tamano de "pip" para reportar costes/distancias en unidades legibles.

    JPY y metales cotizan con 2 decimales (pip=0.01) en vez de 4/5 (pip=0.0001).
    Los indices (US500) cotizan en puntos con 2 decimales -- usar el pip de forex
    (0.0001) ahi es un error de 2 ordenes de magnitud (bug real encontrado el
    2026-08-26: `point` de US500 en MT5 es 0.01, no 0.00001 como un par forex).
    """
    s = symbol.upper()
    if s in _INDEX_SYMBOLS:
        return 0.01
    if s.endswith("JPY") or s in {"XAUUSD", "XAGUSD"}:
        return 0.01
    return 0.0001


def min_sl_distance_price(symbol: str, min_pips: float = 12.0) -> float:
    """Piso minimo absoluto de distancia de SL, en precio, independiente de cuan
    bajo este el ATR momentaneo.

    Motivado por evidencia real (2026-08-27, 3 dias de piloto en vivo): las
    operaciones con SL por debajo de ~8 pips (detectadas por
    `scripts/daily_review.py`) ganaron 1/8 (12.5%) con -$1178.89, contra 8/18
    (44.4%) con +$493.61 en el resto -- sin esas operaciones el piloto hubiera
    estado en positivo en vez de en negativo. El SL calculado por ATR puede quedar
    muy corto cuando la volatilidad momentanea es baja; este piso lo evita.

    Devuelve 0.0 (sin piso) para indices -- "pips" de forex no es una unidad
    dimensionalmente valida para un instrumento que cotiza en puntos de precio muy
    distintos (ver `pip_size`); no hay evidencia todavia de que US500 tenga el
    mismo problema (de hecho nunca aparecio en las operaciones flageadas).
    """
    if symbol.upper() in _INDEX_SYMBOLS:
        return 0.0
    return min_pips * pip_size(symbol)


def compute_sl_tp_with_floor(
    entry_price: float,
    sign: int,
    atr: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
    symbol: str,
    min_sl_pips: float = 12.0,
) -> tuple[float, float]:
    """SL/TP a partir de ATR, con el piso minimo de `min_sl_distance_price` aplicado
    al SL. Si el ATR momentaneo da un SL mas corto que el piso, el TP se escala
    proporcionalmente usando el ratio CONFIGURADO (tp_atr_mult/sl_atr_mult) -- no un
    factor derivado de la distancia original -- para que el risk:reward no se toque,
    y para que siga bien definido incluso si `atr` fuera 0.

    `sign`: +1 para LONG, -1 para SHORT (no se importa `Direction` aca para no
    acoplar este modulo de metadata de instrumentos a los tipos de señal).
    """
    sl_distance = sl_atr_mult * atr
    tp_distance = tp_atr_mult * atr

    floor = min_sl_distance_price(symbol, min_sl_pips)
    if floor > 0 and sl_distance < floor:
        ratio = tp_atr_mult / sl_atr_mult if sl_atr_mult > 0 else 1.0
        sl_distance = floor
        tp_distance = floor * ratio

    take_profit = entry_price + sign * tp_distance
    stop_loss = entry_price - sign * sl_distance
    return stop_loss, take_profit
