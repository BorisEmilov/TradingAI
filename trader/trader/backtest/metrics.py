"""Metrics computed on realized R-multiples (net of the spread cost already
folded into each `TradeResult.net_r`) -- R-multiples instead of $/pip P&L on
purpose, since no position sizing model exists here; R is what's directly
comparable across symbols and consistent with how the whole pipeline already
gates entries (minimum 1:2 R:R).
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.backtest.trade import TradeResult


@dataclass(frozen=True)
class MetricsSummary:
    n_trades: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    max_drawdown_r: float


def compute_metrics(trades: list[TradeResult]) -> MetricsSummary:
    if not trades:
        return MetricsSummary(n_trades=0, win_rate=0.0, expectancy_r=0.0, profit_factor=0.0, max_drawdown_r=0.0)

    rs = [t.net_r for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]

    win_rate = len(wins) / len(rs)
    expectancy_r = sum(rs) / len(rs)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    ordered = sorted(trades, key=lambda t: t.exit_time)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in ordered:
        equity += t.net_r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return MetricsSummary(
        n_trades=len(rs),
        win_rate=win_rate,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        max_drawdown_r=max_dd,
    )


def breakdown_by(trades: list[TradeResult], key_fn) -> dict:
    groups: dict = {}
    for t in trades:
        groups.setdefault(key_fn(t), []).append(t)
    return {key: compute_metrics(group_trades) for key, group_trades in groups.items()}


def breakdown_by_symbol(trades: list[TradeResult]) -> dict:
    return breakdown_by(trades, lambda t: t.symbol)


def breakdown_by_category(trades: list[TradeResult]) -> dict:
    return breakdown_by(trades, lambda t: t.category)


def breakdown_by_symbol_and_category(trades: list[TradeResult]) -> dict:
    return breakdown_by(trades, lambda t: (t.symbol, t.category))


def breakdown_by_session(trades: list[TradeResult]) -> dict:
    def _session_key(t: TradeResult) -> str:
        return "/".join(t.session_labels) if t.session_labels else "ninguna"

    return breakdown_by(trades, _session_key)


def breakdown_by_confluence(trades: list[TradeResult]) -> dict:
    return breakdown_by(trades, lambda t: t.dominant_confluence_family)
