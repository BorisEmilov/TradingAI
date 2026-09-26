"""Open-position management per spec sections 5-6: take partial + move SL to
breakeven at the configured progress threshold; protect capital with an early
close if a higher-timeframe CHoCH invalidates the idea before that; once the
partial is taken, the remainder runs to TP unless structure invalidates, in
which case it's closed at breakeven (the SL is already there by then)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PositionAction(str, Enum):
    HOLD = "hold"
    TAKE_PARTIAL_AND_MOVE_SL_BE = "take_partial_and_move_sl_be"
    CLOSE_BREAKEVEN = "close_breakeven"
    CLOSE_FULL = "close_full"


@dataclass
class PositionPlan:
    direction: str  # "long" | "short"
    entry: float
    sl: float
    tp: float
    partial_taken: bool = False


def progress_to_tp(plan: PositionPlan, current_price: float) -> float:
    total = (plan.tp - plan.entry) if plan.direction == "long" else (plan.entry - plan.tp)
    if total <= 0:
        return 0.0
    moved = (current_price - plan.entry) if plan.direction == "long" else (plan.entry - current_price)
    return moved / total


def evaluate_position(
    plan: PositionPlan,
    current_price: float,
    partial_at_progress_pct: float,
    htf_invalidation: bool,
) -> PositionAction:
    if htf_invalidation:
        return PositionAction.CLOSE_BREAKEVEN if plan.partial_taken else PositionAction.CLOSE_FULL

    progress = progress_to_tp(plan, current_price)
    if not plan.partial_taken and progress >= partial_at_progress_pct:
        return PositionAction.TAKE_PARTIAL_AND_MOVE_SL_BE

    return PositionAction.HOLD
