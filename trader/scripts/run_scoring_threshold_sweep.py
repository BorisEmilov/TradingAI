"""Threshold sweep for the composite weighted score, per
prompt-sistema-puntuacion-ponderada.md step 3 (paired with
prompt-aprobacion-implementar-scoring.md's go-ahead). Detectors run ONCE per
symbol; every threshold combo re-runs cheaply against that same precomputed
data (`run_symbol_backtest_from_analyses`) -- same pattern as
scripts/run_sensitivity_sweep.py.

Only `scoring.min_score_trend`/`min_score_reversal` vary. Everything else
(weights, R:R, SL floor, news filter, M15 window=3) stays exactly as
approved/configured -- this script does not pick a winner, it reports the
full table.

Sweeps the TREND threshold across a spread of values and derives the
REVERSAL threshold from the same 3:4 ratio (trend/0.75) used throughout this
project's confluence-count history, rather than an independent 2D grid --
keeps the one principled relationship (reversal needs more evidence) fixed
while varying overall strictness, and avoids a combinatorial multiple-
comparisons fishing expedition over two free parameters.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.engine import run_symbol_backtest_from_analyses
from trader.backtest.metrics import breakdown_by_category, compute_metrics
from trader.config import TraderConfig, load_config
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.engine import TimeframeAnalysis

TIMEFRAMES = ("D1", "H1", "M15")
DEFAULT_COUNTS = {"D1": 1500, "H1": 20000, "M15": 50000}

TREND_VALUES = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # 50 = design doc's sweep center
REVERSAL_RATIO = 0.75  # trend:reversal 3:4 ratio this replaces, see the weights proposal doc
MAX_SCORE = 90.0


def _variant(base: TraderConfig, *, min_score_trend: float, min_score_reversal: float) -> TraderConfig:
    return dataclasses.replace(
        base, scoring=dataclasses.replace(base.scoring, min_score_trend=min_score_trend, min_score_reversal=min_score_reversal)
    )


def _fmt(m) -> str:
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return f"n={m.n_trades:3d} win_rate={m.win_rate:.1%} expectancy_r={m.expectancy_r:+.3f} pf={pf} max_dd_r={m.max_drawdown_r:.2f}"


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    per_symbol_data = {}
    try:
        for symbol in config.symbols:
            print(f"=== {symbol}: fetching history + running detectors (once) ===")
            candles = {tf: client.candles(symbol, tf, DEFAULT_COUNTS[tf], timeout=120) for tf in TIMEFRAMES}
            analyses = {tf: TimeframeAnalysis.from_candles(candles[tf], tf, config) for tf in TIMEFRAMES}
            info = client.symbol_info(symbol, timeout=30)
            cost = estimate_symbol_cost(symbol, float(info["point"]), candles["M15"])
            per_symbol_data[symbol] = (analyses, candles["M15"], cost)
    finally:
        client.logout()

    combos = []
    for trend in TREND_VALUES:
        reversal = min(trend / REVERSAL_RATIO, MAX_SCORE)
        combos.append((trend, reversal))

    print(f"\n=== corriendo {len(combos)} umbrales sobre {len(config.symbols)} simbolos ===\n")

    results = []
    report_lines = [
        "# Barrido de umbral minimo -- sistema de puntuacion ponderada",
        "",
        "Centro del barrido (punto de partida del diseno, NO el resultado final): "
        f"tendencia=50/{MAX_SCORE:.0f}, reversion=67/{MAX_SCORE:.0f} (razon 3:4={REVERSAL_RATIO}).",
        "",
        "| Umbral tendencia | Umbral reversion | n | win rate | expectancy_r | PF | max_dd_r | n tendencia | n reversion |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for min_score_trend, min_score_reversal in combos:
        variant_config = _variant(config, min_score_trend=min_score_trend, min_score_reversal=min_score_reversal)

        all_trades = []
        all_no_signals = []
        for symbol, (analyses, m15_df, cost) in per_symbol_data.items():
            r = run_symbol_backtest_from_analyses(symbol, analyses, m15_df, variant_config, cost)
            all_trades.extend(r.trades)
            all_no_signals.extend(r.no_signals)

        m = compute_metrics(all_trades)
        by_cat = breakdown_by_category(all_trades)
        n_trend = by_cat.get("trend").n_trades if "trend" in by_cat else 0
        n_reversal = by_cat.get("reversal").n_trades if "reversal" in by_cat else 0

        results.append(
            {
                "min_score_trend": min_score_trend,
                "min_score_reversal": min_score_reversal,
                "n_trades": m.n_trades,
                "win_rate": m.win_rate,
                "expectancy_r": m.expectancy_r,
                "profit_factor": m.profit_factor if m.profit_factor != float("inf") else None,
                "max_drawdown_r": m.max_drawdown_r,
                "n_trend": n_trend,
                "n_reversal": n_reversal,
            }
        )
        pf_str = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        line = (
            f"| {min_score_trend:.0f}/{MAX_SCORE:.0f} | {min_score_reversal:.0f}/{MAX_SCORE:.0f} | {m.n_trades} | "
            f"{m.win_rate:.1%} | {m.expectancy_r:+.3f} | {pf_str} | {m.max_drawdown_r:.2f} | {n_trend} | {n_reversal} |"
        )
        report_lines.append(line)
        print(f"trend>={min_score_trend:.0f} reversal>={min_score_reversal:.0f} -> {_fmt(m)} (trend={n_trend} reversal={n_reversal})")

    logs_dir = Path(config.signals_log_path).parent
    logs_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = logs_dir / "scoring_threshold_sweep.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    report = "\n".join(report_lines) + "\n"
    report_path = logs_dir / "scoring_threshold_sweep_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n{report}")
    print(f"Guardado: {jsonl_path}")
    print(f"Guardado: {report_path}")


if __name__ == "__main__":
    main()
