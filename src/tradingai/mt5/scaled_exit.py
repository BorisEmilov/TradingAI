"""Salida escalonada: cierra una fraccion de la posicion en un primer TP conservador
(a mitad de camino hacia el TP original) y deja el resto corriendo para que lo gestione
el trailing stop basado en estructura (ver tradingai.mt5.trailing_stop).

Con el R:R por defecto de esta config (sl_atr_mult=2.0, tp_atr_mult=4.0, es decir
TP a 2R), un `tp1_fraction=0.5` pone el primer TP exactamente en 1R -- el mismo punto
en el que se activa el trailing (`activate_at_r_multiple=1.0`), asi que al momento de
tomar la ganancia parcial el resto de la posicion ya empieza a protegerse solo, sin
necesidad de mover el SL a breakeven a mano.
"""

from __future__ import annotations

from tradingai.core.signal import Direction


def compute_tp1(entry_price: float, take_profit: float, direction: Direction, tp1_fraction: float = 0.5) -> float:
    """Precio del primer TP: `tp1_fraction` del camino entre la entrada y el TP original."""
    return entry_price + tp1_fraction * (take_profit - entry_price)


def should_take_partial_profit(
    direction: Direction,
    entry_price: float,
    take_profit: float,
    current_price: float,
    tp1_fraction: float = 0.5,
) -> bool:
    """True si el precio ya alcanzo (o supero) el primer TP conservador."""
    if not take_profit:
        return False
    tp1 = compute_tp1(entry_price, take_profit, direction, tp1_fraction)
    if direction == Direction.LONG:
        return current_price >= tp1
    if direction == Direction.SHORT:
        return current_price <= tp1
    return False


def should_move_to_breakeven(direction: Direction, entry_price: float, current_sl: float) -> bool:
    """True si el SL actual esta PEOR que breakeven (ni el trailing ni nada lo movio
    todavia) y hay que forzarlo a la entrada. False si ya esta en breakeven o mejor
    (el trailing ya lo supero con una zona de estructura) -- nunca lo afloja.

    Pensado para usarse SOLO en posiciones que ya tuvieron un cierre parcial (ver
    `should_take_partial_profit`): tras bancar una fraccion en TP1, el resto no
    deberia poder terminar en perdida neta si el trailing todavia no encontro una
    zona mejor donde proteger el SL.
    """
    if direction == Direction.LONG:
        return current_sl < entry_price
    if direction == Direction.SHORT:
        return current_sl > entry_price
    return False
