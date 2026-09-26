"""Parte 2 de prompt-auditoria-detectores-y-desglose.md: breakdown of the most
recent valid backtest (SL floor fix applied) by symbol and year -- to see
whether the scarcity of trades is uniform over time or concentrated.

Reads the CURRENT logs/backtest_trades.jsonl and logs/backtest_no_signals.jsonl
(written by the last scripts/run_backtest.py run) -- does not re-run the
backtest itself. Writes logs/trades_by_symbol_year_report.md.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.config import load_config


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    config = load_config()
    logs_dir = Path(config.signals_log_path).parent

    trades = _load_jsonl(logs_dir / "backtest_trades.jsonl")
    no_signals = _load_jsonl(logs_dir / "backtest_no_signals.jsonl")

    if not trades and not no_signals:
        print("No hay logs/backtest_trades.jsonl ni logs/backtest_no_signals.jsonl -- corre scripts/run_backtest.py primero.")
        return

    for t in trades:
        t["_year"] = pd.Timestamp(t["entry_time"]).year
    for n in no_signals:
        n["_year"] = pd.Timestamp(n["as_of"]).year

    symbols = sorted({t["symbol"] for t in trades} | {n["symbol"] for n in no_signals})
    years = sorted({t["_year"] for t in trades} | {n["_year"] for n in no_signals})

    trades_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for t in trades:
        trades_by_key[(t["symbol"], t["_year"])].append(t)

    evaluated_by_key: dict[tuple[str, int], int] = defaultdict(int)
    for n in no_signals:
        evaluated_by_key[(n["symbol"], n["_year"])] += 1
    for t in trades:
        evaluated_by_key[(t["symbol"], t["_year"])] += 1

    lines = [
        "# Desglose de operaciones por simbolo y anio",
        "",
        f"Backtest fuente: logs/backtest_trades.jsonl + logs/backtest_no_signals.jsonl "
        f"(total {len(trades)} operaciones, {len(no_signals)} setups evaluados y descartados).",
        "",
        "## Operaciones ejecutadas por simbolo y anio",
        "",
        "| Simbolo | " + " | ".join(str(y) for y in years) + " | Total |",
        "|---|" + "---|" * (len(years) + 1),
    ]
    for symbol in symbols:
        row = [symbol]
        total = 0
        for y in years:
            n = len(trades_by_key.get((symbol, y), []))
            total += n
            row.append(str(n) if n else "-")
        row.append(str(total))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Candidatos evaluados (con y sin operacion final) por simbolo y anio", ""]
    lines += ["| Simbolo | " + " | ".join(str(y) for y in years) + " | Total |", "|---|" + "---|" * (len(years) + 1)]
    for symbol in symbols:
        row = [symbol]
        total = 0
        for y in years:
            n = evaluated_by_key.get((symbol, y), 0)
            total += n
            row.append(str(n) if n else "-")
        row.append(str(total))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Metricas por simbolo y anio (solo donde hay operaciones)",
        "",
        "n bajo (<5) no es significativo estadisticamente -- se reporta igual, marcado.",
        "",
        "| Simbolo | Anio | n | Win rate | Expectancy (R) | Profit factor |",
        "|---|---|---|---|---|---|",
    ]
    any_trades = False
    for symbol in symbols:
        for y in years:
            group = trades_by_key.get((symbol, y), [])
            if not group:
                continue
            any_trades = True
            rs = [t["net_r"] for t in group]
            wins = [r for r in rs if r > 0]
            losses = [r for r in rs if r <= 0]
            win_rate = len(wins) / len(rs)
            expectancy = sum(rs) / len(rs)
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            flag = " (n<5, no significativo)" if len(rs) < 5 else ""
            lines.append(f"| {symbol} | {y} | {len(rs)}{flag} | {win_rate:.1%} | {expectancy:+.3f} | {pf_str} |")
    if not any_trades:
        lines.append("| -- ninguna combinacion simbolo/anio tiene operaciones -- |  |  |  |  |  |")

    report = "\n".join(lines) + "\n"
    out_path = logs_dir / "trades_by_symbol_year_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
