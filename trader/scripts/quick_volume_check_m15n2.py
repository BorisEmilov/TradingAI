"""Step 5 of prompt-implementar-m15-n2.md: quick single-symbol volume check
BEFORE investing time in the full 3-symbol/8-year backtest -- confirms the
order of magnitude the audit predicted (M15 window_candles=2 + news filter)
actually shows up in a real, full-pipeline run, not just in the isolated
M15_confirmation x H1_poi audit.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.engine import run_symbol_backtest
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

SYMBOL = "EURUSD"
COUNTS = {"D1": 1500, "H1": 20000, "M15": 50000}


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        candles = {tf: client.candles(SYMBOL, tf, COUNTS[tf], timeout=120) for tf in ("D1", "H1", "M15")}
        info = client.symbol_info(SYMBOL, timeout=30)
        cost = estimate_symbol_cost(SYMBOL, float(info["point"]), candles["M15"])
    finally:
        client.logout()

    for tf, df in candles.items():
        print(f"{tf}: {len(df)} velas, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")

    print("\n=== window_candles=1, news_filter OFF (baseline de referencia, fix OB/FVG solo) ===")
    cfg_baseline = replace(
        config,
        m15_confirmation=replace(config.m15_confirmation, window_candles=1),
        news_filter=replace(config.news_filter, enabled=False),
    )
    result_baseline = run_symbol_backtest(SYMBOL, candles, cfg_baseline, cost)
    print(f"{len(result_baseline.trades)} operaciones, {len(result_baseline.no_signals)} descartados")
    print(Counter((n.stage, n.reason) for n in result_baseline.no_signals).most_common(10))

    print("\n=== window_candles=2, news_filter ON (config.yaml actual -- lo que se va a backtest-ear completo) ===")
    result_new = run_symbol_backtest(SYMBOL, candles, config, cost)
    print(f"{len(result_new.trades)} operaciones, {len(result_new.no_signals)} descartados")
    print(Counter((n.stage, n.reason) for n in result_new.no_signals).most_common(10))


if __name__ == "__main__":
    main()
