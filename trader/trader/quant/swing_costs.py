"""prompt-fase-cuantitativa.md (pivote swing/posicion): cost model extension for
multi-day holds -- spread alone (trader/backtest/costs.py) is what the intraday
system needed since it never held overnight. Two things that only exist once a
position can span days/weeks:

1. Swap/rollover: MT5 charges (or, rarely, credits) financing per calendar day
   held, in POINTS per lot (this account's symbols are all `swap_mode=1`,
   POINTS-based -- verified via a real /symbols/{symbol} call, not assumed).
   Wednesday carries triple swap on this account (`swap_rollover3days=3`,
   verified per-symbol, not assumed to be the textbook-standard Wednesday).
2. Weekend gap risk: a stop-loss can't fill at its nominal price if Monday's
   open is already through it -- something an intraday system that always
   closed same-day never had to model. Measured directly from real D1
   open/close data (no fabricated distribution).

Known simplification, documented not hidden (same discipline as
backtest/costs.py): swap rates are fetched ONCE, today -- there's no
historical swap-rate time series available from this data source, so every
backtest day uses TODAY's rate as a constant proxy for the whole 1999-2026
window. Real rate differentials moved a lot over that period (near-zero
2009-2021, hiking cycle 2022+) -- this likely understates swap cost in the
high-rate years and overstates it in the low-rate years. Flagged in the
battery report, not silently absorbed into a single number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwapInfo:
    symbol: str
    point: float
    swap_long_points: float
    swap_short_points: float
    rollover_triple_weekday: int  # Python weekday() convention: Monday=0..Sunday=6


def load_swap_info(symbol: str, raw_symbol_info: dict) -> SwapInfo:
    # MT5's day-of-week for swap_rollover3days uses Sunday=0..Saturday=6;
    # Python's datetime.weekday() uses Monday=0..Sunday=6 -- convert once here
    # so callers never have to remember two different day-of-week conventions.
    mt5_dow = int(raw_symbol_info["swap_rollover3days"])
    python_dow = (mt5_dow - 1) % 7
    return SwapInfo(
        symbol=symbol,
        point=float(raw_symbol_info["point"]),
        swap_long_points=float(raw_symbol_info["swap_long"]),
        swap_short_points=float(raw_symbol_info["swap_short"]),
        rollover_triple_weekday=python_dow,
    )


def swap_cost_frac(direction: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp, swap: SwapInfo, mean_price: float) -> float:
    """Total swap cost, as a fraction of price (log-return-comparable, same
    convention as backtest/costs.py's spread fraction), for a position held
    from `entry_date` (exclusive -- the day it was opened doesn't accrue an
    overnight charge yet) through `exit_date` (inclusive of the last night
    held). Always a COST here (`abs`) -- this account's swap_long/swap_short
    are both negative on all 3 symbols (broker markup dominates whatever the
    underlying rate differential is), so there's no direction where holding
    is free; modeling it as strictly a drag matches the real quoted numbers,
    not an assumption.
    """
    points_per_day = abs(swap.swap_long_points if direction == "long" else swap.swap_short_points)
    frac_per_day = points_per_day * swap.point / mean_price

    nights = pd.date_range(entry_date.normalize() + pd.Timedelta(days=1), exit_date.normalize(), freq="D")
    if len(nights) == 0:
        return 0.0
    multiplier = np.where(nights.weekday == swap.rollover_triple_weekday, 3, 1)
    return float(frac_per_day * multiplier.sum())


def weekend_gap_stats(candles_d1: pd.DataFrame, min_calendar_gap_days: int = 3) -> dict:
    """Real open-vs-prior-close gap distribution across non-trading-day breaks
    (weekends, and incidentally holidays >=3 calendar days) -- measured, not
    modeled from a parametric assumption."""
    ts = candles_d1["timestamp"]
    gap_days = ts.diff().dt.days
    is_break = (gap_days >= min_calendar_gap_days).to_numpy()

    log_open = np.log(candles_d1["open"].to_numpy())
    log_close_prev = np.log(candles_d1["close"].shift(1).to_numpy())
    gap_return = log_open - log_close_prev
    gaps = gap_return[is_break]
    gaps = gaps[~np.isnan(gaps)]

    if len(gaps) == 0:
        return {"n": 0}
    return {
        "n": int(len(gaps)),
        "mean": float(gaps.mean()),
        "std": float(gaps.std()),
        "p1": float(np.percentile(gaps, 1)),
        "p5": float(np.percentile(gaps, 5)),
        "p95": float(np.percentile(gaps, 95)),
        "p99": float(np.percentile(gaps, 99)),
        "worst": float(gaps.min()),
        "best": float(gaps.max()),
    }
