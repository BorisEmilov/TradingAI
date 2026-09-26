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
