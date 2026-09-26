"""Paso 1-2 de prompt-fix-fvg-minimo-ltf.md: deriva un umbral minimo de
tamano de FVG (min_gap_pct) para las variantes M15/M30 del experimento
D1-LTF, anclado en datos reales de spread y ATR de los 3 simbolos (no a ojo).

Tambien recalcula, con el umbral propuesto, que fraccion de las FVG
actualmente elegibles (10131 en M15 / 4915 en M30 para EURUSD, del chequeo
de volumen anterior) caen por debajo -- para los 3 simbolos.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.backtest.costs import estimate_symbol_cost
from trader.config import load_config
from trader.detectors.fvg import detect_fvg
from trader.detectors.indicators import atr as atr_fn
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.experiment_ltf import LTF_15M, LTF_30M

SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY")
M15_COUNT = 50000
M30_COUNT = 25000


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    per_symbol = {}
    try:
        for symbol in SYMBOLS:
            m15_df = client.candles(symbol, "M15", M15_COUNT, timeout=120)
            m30_df = client.candles(symbol, "M30", M30_COUNT, timeout=120)
            info = client.symbol_info(symbol, timeout=30)
            per_symbol[symbol] = (m15_df, m30_df, float(info["point"]))
    finally:
        client.logout()

    print("=== spread real y ATR por simbolo/timeframe ===\n")
    stats = {}
    for symbol in SYMBOLS:
        m15_df, m30_df, point = per_symbol[symbol]
        pip = point * 10 if "JPY" not in symbol else point * 10  # both cases: 1 pip = 10 points for these symbols

        for tf_label, df in [("M15", m15_df), ("M30", m30_df)]:
            cost = estimate_symbol_cost(symbol, point, df)
            avg_price = float(df["close"].mean())
            spread_price = cost.avg_spread_price
            spread_pct = spread_price / avg_price * 100.0
            spread_pips = spread_price / pip

            atr_series = atr_fn(df, period=14)
            avg_atr = float(atr_series.dropna().mean())
            atr_pct = avg_atr / avg_price * 100.0
            atr_pips = avg_atr / pip

            stats[(symbol, tf_label)] = dict(
                avg_price=avg_price, spread_price=spread_price, spread_pct=spread_pct, spread_pips=spread_pips,
                avg_atr=avg_atr, atr_pct=atr_pct, atr_pips=atr_pips,
            )
            print(f"{symbol} {tf_label}: precio_prom={avg_price:.5f} spread={spread_pips:.2f} pips ({spread_pct:.4f}%) "
                  f"ATR={atr_pips:.2f} pips ({atr_pct:.4f}%)")

    print("\n=== umbral propuesto ===\n")
    for tf_label in ("M15", "M30"):
        max_spread_pct = max(stats[(s, tf_label)]["spread_pct"] for s in SYMBOLS)
        widest_symbol = max(SYMBOLS, key=lambda s: stats[(s, tf_label)]["spread_pct"])
        for margin in (2.0, 3.0):
            proposed = max_spread_pct * margin
            print(f"{tf_label}: margen {margin}x sobre el spread mas ancho ({widest_symbol}, {max_spread_pct:.4f}%) -> min_gap_pct={proposed:.4f}%")

    print("\n=== impacto: cuantas FVG actualmente elegibles caen bajo el nuevo umbral ===\n")
    for thresholds, tf_label in [(LTF_15M, "M15"), (LTF_30M, "M30")]:
        max_spread_pct = max(stats[(s, tf_label)]["spread_pct"] for s in SYMBOLS)
        for margin in (2.0, 3.0):
            proposed = max_spread_pct * margin
            for symbol in SYMBOLS:
                df = per_symbol[symbol][0] if tf_label == "M15" else per_symbol[symbol][1]
                raw_current = detect_fvg(df, tf_label, min_gap_pct=0.0)
                raw_proposed = detect_fvg(df, tf_label, min_gap_pct=proposed)
                dropped = len(raw_current) - len(raw_proposed)
                pct_dropped = 100 * dropped / len(raw_current) if raw_current else 0.0
                print(f"  margen={margin}x min_gap_pct={proposed:.4f}% {symbol} {tf_label}: "
                      f"{len(raw_current)} -> {len(raw_proposed)} ({dropped} descartadas, {pct_dropped:.1f}%)")


if __name__ == "__main__":
    main()
