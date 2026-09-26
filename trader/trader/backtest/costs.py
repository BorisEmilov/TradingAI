"""Per-symbol trading cost, estimated from real data instead of a guessed
constant: the average spread actually observed in the historical M15 candles
(MT5 reports `spread`, in points, per bar) converted to price via the
symbol's real `point` size from `/symbols/{symbol}`.

CAVEAT (verificado 2026-09-25, scripts/verify_spread_source.py): el campo
`spread` de una vela MT5 es el spread MÍNIMO dentro de esa vela, no el
promedio (320/320 velas iguales al mínimo de sus ticks). Esto SUBESTIMA el
costo real: may-2026 EURUSD mín 0.7 vs promedio de ticks 5.2 pts; desde
jul-2026 el mínimo es ~0 en casi todas las velas. Los números de backtest que
salen de acá son una cota inferior del costo. El piloto MTF en vivo ya no usa
esto: lee `PythonGetawayClient.recent_avg_spread` (ticks reales).

Only spread is modeled -- no slippage or commission, since neither is
available from this data source. Documented as a known simplification, not
hidden: `docstring`/README call this out so the backtest numbers aren't
mistaken for a complete cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SymbolCost:
    symbol: str
    point: float
    avg_spread_points: float
    avg_spread_price: float


def estimate_symbol_cost(symbol: str, point: float, m15_df: pd.DataFrame) -> SymbolCost:
    if "spread" in m15_df.columns and len(m15_df) > 0:
        avg_spread_points = float(m15_df["spread"].mean())
    else:
        avg_spread_points = 0.0
    return SymbolCost(
        symbol=symbol,
        point=point,
        avg_spread_points=avg_spread_points,
        avg_spread_price=avg_spread_points * point,
    )


def cost_in_r(cost: SymbolCost, risk_price: float) -> float:
    """Round-trip spread cost expressed in R (fraction of the risked distance)."""
    if risk_price <= 0:
        return 0.0
    return cost.avg_spread_price / risk_price
