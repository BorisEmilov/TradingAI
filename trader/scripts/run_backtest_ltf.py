"""Paso 5 de prompt-experimento-d1-30m-15m.md (con el fix de
prompt-fix-fvg-minimo-ltf.md ya aplicado): backtest completo de 3 simbolos
para las variantes D1->M15-unico y D1->M30-unico, mismo periodo que todos
los baselines anteriores.

CONFUSOR CONOCIDO, sin resolver en este script (per instruccion 6 de
prompt-fix-fvg-minimo-ltf.md): el sistema de scoring (min_score_trend=50/90,
min_score_reversal=67/90) se calibro pensando en la jerarquia H1->M15 de dos
etapas -- ejes como "confluencia HTF" o "calidad de zona" pueden pesar
distinto cuando la misma vela hace de zona Y de gatillo. `score_below_minimum`
fue 23-26% del rechazo en el chequeo de volumen -- si el backtest completo no
muestra mejora sustancial, NO se puede concluir "el timeframe unico no ayuda"
sin antes descartar que el score viejo este ahogando el experimento.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.metrics import breakdown_by_category, breakdown_by_symbol_and_category, compute_metrics
from trader.backtest.trade import TradeResult
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.experiment_ltf import LTF_15M, LTF_30M, LtfThresholds, build_ltf_analysis, run_symbol_backtest_ltf

D1_COUNT = 1500
M15_COUNT = 50000
M30_COUNT = 25000


def _json_default(o):
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _fmt(m) -> str:
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return f"n={m.n_trades}, win_rate={m.win_rate:.1%}, expectancy_r={m.expectancy_r:+.3f}, profit_factor={pf}, max_dd_r={m.max_drawdown_r:.2f}"


def run_variant(thresholds: LtfThresholds, config, client: PythonGetawayClient) -> tuple[list[TradeResult], list]:
    all_trades: list[TradeResult] = []
    all_no_signals = []
    ltf_count = M15_COUNT if thresholds.timeframe == "M15" else M30_COUNT

    for symbol in config.symbols:
        print(f"=== {thresholds.timeframe} {symbol}: fetching history ===")
        d1_df = client.candles(symbol, "D1", D1_COUNT, timeout=120)
        ltf_df = client.candles(symbol, thresholds.timeframe, ltf_count, timeout=120)
        print(f"  D1: {len(d1_df)} velas, {thresholds.timeframe}: {len(ltf_df)} velas, "
              f"{ltf_df['timestamp'].iloc[0].date()} -> {ltf_df['timestamp'].iloc[-1].date()}")

        info = client.symbol_info(symbol, timeout=30)
        cost = estimate_symbol_cost(symbol, float(info["point"]), ltf_df)

        d1_analysis = TimeframeAnalysis.from_candles(d1_df, "D1", config)
        ltf_analysis = build_ltf_analysis(ltf_df, thresholds, config)

        print(f"=== {thresholds.timeframe} {symbol}: corriendo backtest ===")
        result = run_symbol_backtest_ltf(symbol, d1_analysis, ltf_analysis, ltf_df, thresholds, config, cost)
        print(f"  {len(result.trades)} operaciones, {len(result.no_signals)} descartados")

        all_trades.extend(result.trades)
        all_no_signals.extend(result.no_signals)

    return all_trades, all_no_signals


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    logs_dir = Path(config.signals_log_path).parent
    report_lines = ["# Backtest D1->LTF-unico (M15 y M30) -- experimento aislado", ""]

    try:
        for thresholds in (LTF_15M, LTF_30M):
            trades, no_signals = run_variant(thresholds, config, client)

            trades_path = logs_dir / f"backtest_trades_ltf_{thresholds.timeframe.lower()}.jsonl"
            no_signals_path = logs_dir / f"backtest_no_signals_ltf_{thresholds.timeframe.lower()}.jsonl"
            with trades_path.open("w", encoding="utf-8") as fh:
                for t in trades:
                    fh.write(json.dumps(asdict(t), default=_json_default) + "\n")
            with no_signals_path.open("w", encoding="utf-8") as fh:
                for n in no_signals:
                    fh.write(json.dumps(asdict(n), default=_json_default) + "\n")

            m = compute_metrics(trades)
            by_cat = breakdown_by_category(trades)
            by_sym_cat = breakdown_by_symbol_and_category(trades)

            print(f"\n=== {thresholds.timeframe} agregado: {_fmt(m)} ===")
            report_lines += [f"## {thresholds.timeframe}", "", f"Total operaciones: {len(trades)}, setups evaluados: {len(trades) + len(no_signals)}", "", f"Agregado: {_fmt(m)}", ""]
            report_lines += ["### Por categoria", ""]
            for cat, cm in by_cat.items():
                report_lines.append(f"- {cat}: {_fmt(cm)}")
            report_lines += ["", "### Por simbolo y categoria", ""]
            for (sym, cat), cm in by_sym_cat.items():
                report_lines.append(f"- {sym} / {cat}: {_fmt(cm)}")
            report_lines.append("")
    finally:
        client.logout()

    report = "\n".join(report_lines) + "\n"
    report_path = logs_dir / "backtest_ltf_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nGuardado: {report_path}")


if __name__ == "__main__":
    main()
