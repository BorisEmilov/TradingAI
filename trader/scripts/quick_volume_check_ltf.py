"""Paso 3 de prompt-experimento-d1-30m-15m.md: chequeo rapido de volumen para
las variantes D1->M15-unico y D1->M30-unico, ANTES de invertir tiempo en el
backtest completo de 3 simbolos. Compara contra el baseline de produccion
(H1_poi rechazaba ~38-39% de los candidatos que llegaban a esa etapa).

Un solo simbolo (EURUSD), escala completa (mismos counts que
scripts/run_backtest.py), para tener una lectura real del orden de magnitud.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.backtest.costs import estimate_symbol_cost
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.experiment_ltf import LTF_15M, LTF_30M, build_ltf_analysis, run_symbol_backtest_ltf

SYMBOL = "EURUSD"
D1_COUNT = 1500
M15_COUNT = 50000
M30_COUNT = 25000  # same real-time span as 50000 M15 candles


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        d1_df = client.candles(SYMBOL, "D1", D1_COUNT, timeout=120)
        m15_df = client.candles(SYMBOL, "M15", M15_COUNT, timeout=120)
        m30_df = client.candles(SYMBOL, "M30", M30_COUNT, timeout=120)
        info = client.symbol_info(SYMBOL, timeout=30)
    finally:
        client.logout()

    cost = estimate_symbol_cost(SYMBOL, float(info["point"]), m15_df)
    d1_analysis = TimeframeAnalysis.from_candles(d1_df, "D1", config)

    print(f"D1: {len(d1_df)} velas, {d1_df['timestamp'].iloc[0].date()} -> {d1_df['timestamp'].iloc[-1].date()}")

    for thresholds, df in [(LTF_15M, m15_df), (LTF_30M, m30_df)]:
        print(f"\n=== {thresholds.timeframe} ===")
        print(f"{thresholds.timeframe}: {len(df)} velas, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")
        ltf_analysis = build_ltf_analysis(df, thresholds, config)
        print(f"OB={len(ltf_analysis.order_blocks)} FVG={len(ltf_analysis.fvgs)} (elegibles: confirmed_at is not None -> "
              f"OB={sum(1 for z in ltf_analysis.order_blocks if z.confirmed_at is not None)} "
              f"FVG={sum(1 for z in ltf_analysis.fvgs if z.confirmed_at is not None)})")

        result = run_symbol_backtest_ltf(SYMBOL, d1_analysis, ltf_analysis, df, thresholds, config, cost)
        total = len(result.trades) + len(result.no_signals)
        print(f"{len(result.trades)} operaciones, {len(result.no_signals)} descartados, {total} evaluados")

        c = Counter((n.stage, n.reason) for n in result.no_signals)
        for k, v in c.most_common(15):
            print(f"  {k}: {v} ({100*v/total:.1f}%)")

        poi_rejects = sum(v for (stage, _), v in c.items() if stage == "LTF_poi")
        print(f"  LTF_poi rechaza {100*poi_rejects/total:.1f}% (baseline H1_poi: ~38-39%)")


if __name__ == "__main__":
    main()
