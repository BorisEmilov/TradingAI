"""SL/TP computed from structure, never from a fixed pip distance -- the
caller passes in the structural stop/target price (from a POI boundary or a
swing/liquidity level). The minimum R:R gate is unconditional: rr < min_rr
returns None, no exceptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeLevels:
    direction: str  # "long" | "short"
    entry: float
    sl: float
    tp: float
    risk_reward: float


def compute_trade_levels(
    direction: str,
    entry: float,
    structural_stop: float,
    structural_target: float,
    min_rr: float,
) -> TradeLevels | None:
    if direction not in ("long", "short"):
        raise ValueError(f"invalid direction: {direction}")

    if direction == "long":
        risk = entry - structural_stop
        reward = structural_target - entry
    else:
        risk = structural_stop - entry
        reward = entry - structural_target

    if risk <= 0 or reward <= 0:
        return None

    rr = reward / risk
    if rr < min_rr:
        return None

    return TradeLevels(direction=direction, entry=entry, sl=structural_stop, tp=structural_target, risk_reward=rr)


def limit_entry_still_ahead(direction: str, price: float, entry: float, tp2: float) -> bool:
    """Una orden límite solo es colocable si el precio de mercado todavía NO
    llegó a la entrada (short: precio < entrada; long: precio > entrada) y
    tampoco pasó el TP2. Es la MISMA geometría que invalida una orden
    pendiente (`compute_trade_levels(..., min_rr=0)` con el SL como frontera),
    usando la entrada límite como frontera: si el precio ya la cruzó, MT5
    rechaza la orden con INVALID_PRICE (GBPJPY 2026-09-29)."""
    return compute_trade_levels(direction, price, entry, tp2, min_rr=0.0) is not None
