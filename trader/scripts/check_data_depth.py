"""Fase 4 step 1: check how much real history MetaQuotes-Demo actually offers
per symbol/timeframe before building anything on top of it. Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

TIMEFRAMES = ["D1", "H1", "M15"]  # H4 removed from the pipeline, see prompt-eliminar-gate-d1-h4.md
MAX_REQUEST = 50000


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        for symbol in config.symbols:
            print(f"\n=== {symbol} ===")
            for tf in TIMEFRAMES:
                try:
                    df = client.candles(symbol, tf, MAX_REQUEST, timeout=120)
                except Exception as exc:  # noqa: BLE001 -- one bad symbol/tf shouldn't kill the whole survey
                    body = getattr(getattr(exc, "response", None), "text", str(exc))
                    print(f"  {tf}: ERROR -- {body[:300]}")
                    continue
                if len(df) == 0:
                    print(f"  {tf}: sin datos")
                    continue
                start, end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
                span_days = (end - start).days
                print(f"  {tf}: {len(df)} velas, {start.date()} -> {end.date()} (~{span_days} dias, ~{span_days/365.25:.1f} anios)")
    finally:
        client.logout()


if __name__ == "__main__":
    main()
