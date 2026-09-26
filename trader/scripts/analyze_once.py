"""Run one multi-timeframe analysis pass for every configured symbol and print
the result (a full signal report, or the reason no signal fired).

This is READ-ONLY: it logs in, fetches candles, runs the pipeline, prints and
logs the outcome. It never calls open_position/close_position/modify_sl --
executing on a signal is a deliberate future step, not something this script
does on its own.

Requires PYGW_LOGIN / PYGW_PASSWORD / PYGW_SERVER in the environment and the
PythonGetaway gateway running (../PythonGetaway/scripts/start.sh).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.logging_ import log_no_signal, log_signal
from trader.pipeline.engine import MultiTimeframePipeline, NoSignal


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    pipeline = MultiTimeframePipeline(config)

    try:
        for symbol in config.symbols:
            candles = client.fetch_multi_timeframe(symbol)
            current_price = client.last_price(symbol)
            as_of = candles["M15"]["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)

            result = pipeline.run(symbol, candles, as_of=as_of, current_price=current_price)

            if isinstance(result, NoSignal):
                print(f"[{symbol}] no signal -- stage={result.stage} reason={result.reason}")
                log_no_signal(result, symbol, as_of, config.signals_log_path)
            else:
                print(result.to_report())
                log_signal(result, config.signals_log_path)
    finally:
        client.logout()


if __name__ == "__main__":
    main()
