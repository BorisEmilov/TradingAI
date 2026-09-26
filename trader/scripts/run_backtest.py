"""Fase 4 deliverable: reproducible backtest of the multi-timeframe pipeline
over real MetaQuotes-Demo history for every configured symbol.

Writes:
  - logs/backtest_trades.jsonl  -- one line per simulated trade, full detail
  - logs/backtest_no_signals.jsonl -- one line per evaluated-but-rejected setup
  - logs/backtest_report.md -- aggregated + per-symbol/session/category/confluence metrics

Usage: python scripts/run_backtest.py [--count 50000]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.engine import NoSignalRecord, run_symbol_backtest
from trader.backtest.metrics import (
    breakdown_by_category,
    breakdown_by_confluence,
    breakdown_by_session,
    breakdown_by_symbol,
    compute_metrics,
)
from trader.backtest.trade import TradeResult
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

TIMEFRAMES = ("D1", "H1", "M15")  # H4 removed from the pipeline, see prompt-eliminar-gate-d1-h4.md
DEFAULT_COUNTS = {"D1": 1500, "H1": 20000, "M15": 50000}

# From the H4-gated baseline run (prompt-fase4-backtesting.md, this same account/
# symbols/period): of 31217 M15 candidates evaluated, 13528 (43%) were rejected
# specifically for D1/H4 bias disagreement, meaning 17689 got PAST that gate to
# reach H1 POI search or later. There is no equivalent gate anymore -- printed
# below as the "quick check" this phase's instructions asked for before
# committing to the full backtest.
BASELINE_TOTAL_EVALUATED = 31217
BASELINE_REACHED_H1_POI_OR_LATER = 31217 - 13528


def _json_default(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), default=_json_default) + "\n")


def _fmt_metrics(m) -> str:
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return f"n={m.n_trades}, win_rate={m.win_rate:.1%}, expectancy_r={m.expectancy_r:+.3f}, profit_factor={pf}, max_dd_r={m.max_drawdown_r:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None, help="override: same count for every timeframe")
    args = parser.parse_args()

    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    all_trades: list[TradeResult] = []
    all_no_signals: list[NoSignalRecord] = []

    try:
        for symbol in config.symbols:
            print(f"=== {symbol}: fetching history ===")
            candles = {
                tf: client.candles(symbol, tf, args.count or DEFAULT_COUNTS[tf], timeout=120) for tf in TIMEFRAMES
            }
            for tf, df in candles.items():
                print(f"  {tf}: {len(df)} velas, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")

            info = client.symbol_info(symbol, timeout=30)
            point = float(info["point"])
            cost = estimate_symbol_cost(symbol, point, candles["M15"])
            print(f"  costo estimado: avg_spread={cost.avg_spread_points:.1f} pts = {cost.avg_spread_price:.6f} precio")

            print(f"=== {symbol}: corriendo backtest ===")
            result = run_symbol_backtest(symbol, candles, config, cost)
            print(f"  {len(result.trades)} operaciones simuladas, {len(result.no_signals)} setups evaluados y descartados")

            all_trades.extend(result.trades)
            all_no_signals.extend(result.no_signals)
    finally:
        client.logout()

    logs_dir = Path(config.signals_log_path).parent
    _write_jsonl(all_trades, logs_dir / "backtest_trades.jsonl")
    _write_jsonl(all_no_signals, logs_dir / "backtest_no_signals.jsonl")

    total_evaluated = len(all_trades) + len(all_no_signals)
    blocked_early = sum(1 for n in all_no_signals if n.stage in ("D1_bias", "session"))
    reached_h1_poi_or_later = total_evaluated - blocked_early

    overall = compute_metrics(all_trades)
    by_symbol = breakdown_by_symbol(all_trades)
    by_session = breakdown_by_session(all_trades)
    by_confluence = breakdown_by_confluence(all_trades)
    by_category = breakdown_by_category(all_trades)

    lines = [
        "# Reporte de backtest -- H4 eliminado, con operativa de reversion",
        "",
        f"Simbolos: {', '.join(config.symbols)}",
        f"Total operaciones simuladas: {len(all_trades)}",
        f"Total setups evaluados y descartados: {len(all_no_signals)}",
        "",
        "## Chequeo rapido: senales desbloqueadas vs. el baseline con H4",
        "",
        f"Baseline (con gate D1/H4): {BASELINE_REACHED_H1_POI_OR_LATER} de {BASELINE_TOTAL_EVALUATED} candidatos "
        f"llegaban a busqueda de POI H1 o mas alla (43% se rechazaba por desalineacion D1/H4).",
        f"Ahora (sin H4): {reached_h1_poi_or_later} de {total_evaluated} candidatos llegan a busqueda de POI H1 o mas alla "
        "(no existe ya ese gate -- solo D1_bias/session bloquean antes de POI).",
        "",
        "## Agregado",
        "",
        _fmt_metrics(overall),
        "",
        "## Por categoria (tendencia vs reversion)",
        "",
    ]
    for category, m in sorted(by_category.items()):
        lines.append(f"- {category}: {_fmt_metrics(m)}")
    lines += ["", "## Por simbolo", ""]
    for symbol, m in sorted(by_symbol.items()):
        lines.append(f"- {symbol}: {_fmt_metrics(m)}")
    lines += ["", "## Por simbolo y categoria", ""]
    for symbol in sorted(config.symbols):
        for category in ("trend", "reversal"):
            subset = [t for t in all_trades if t.symbol == symbol and t.category == category]
            if subset:
                lines.append(f"- {symbol} / {category}: {_fmt_metrics(compute_metrics(subset))}")
    lines += ["", "## Por sesion", ""]
    for session, m in sorted(by_session.items()):
        lines.append(f"- {session}: {_fmt_metrics(m)}")
    lines += ["", "## Por confluencia dominante", ""]
    for family, m in sorted(by_confluence.items()):
        lines.append(f"- {family}: {_fmt_metrics(m)}")

    report = "\n".join(lines) + "\n"
    (logs_dir / "backtest_report.md").write_text(report, encoding="utf-8")

    print()
    print(report)


if __name__ == "__main__":
    main()
