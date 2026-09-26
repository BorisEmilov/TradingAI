"""prompt-fase-cuantitativa.md: fetch + cache real candle data once, for the whole
quant-factor battery, in a SINGLE login/logout gateway session (never one login per
script -- see the project's throttling incident). Also serves as the real depth
check (Fase 4 step 1, `check_data_depth.py`'s read-only survey) since requesting
the same MAX_REQUEST and inspecting what actually comes back IS the depth check.

Writes trader/data/quant_battery/{SYMBOL}_{TIMEFRAME}.csv (timestamp/open/high/low/
close/volume/spread) and prints a depth summary table used to decide the D1
walk-forward-vs-3-way-split question.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

TIMEFRAMES = ["D1", "H1", "M15"]
MAX_REQUEST = 50000
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    print("=== login OK, single session for all fetches ===")

    rows = []
    try:
        for symbol in config.symbols:
            for tf in TIMEFRAMES:
                df = client.candles(symbol, tf, MAX_REQUEST, timeout=120)
                out_path = OUT_DIR / f"{symbol}_{tf}.csv"
                df.to_csv(out_path, index=False)
                if len(df) == 0:
                    rows.append((symbol, tf, 0, None, None, None))
                    print(f"{symbol} {tf}: sin datos")
                    continue
                start, end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
                span_days = (end - start).days
                rows.append((symbol, tf, len(df), start, end, span_days))
                print(f"{symbol} {tf}: {len(df)} velas, {start.date()} -> {end.date()} (~{span_days} dias, ~{span_days/365.25:.2f} anios) -> {out_path}")
    finally:
        client.logout()
        print("=== logout OK ===")

    print("\n=== RESUMEN DE PROFUNDIDAD REAL ===")
    print(f"{'symbol':<8} {'tf':<4} {'n':>7}  {'start':<12} {'end':<12} {'years':>6}")
    for symbol, tf, n, start, end, span_days in rows:
        s = start.date().isoformat() if start is not None else "-"
        e = end.date().isoformat() if end is not None else "-"
        y = f"{span_days/365.25:.2f}" if span_days is not None else "-"
        print(f"{symbol:<8} {tf:<4} {n:>7}  {s:<12} {e:<12} {y:>6}")


if __name__ == "__main__":
    main()
