"""Fase 1 smoke test (prompt-implementacion-agente-trading.md): connect to the
gateway with the configured credentials and print the last N H1 candles of
EURUSD. Read-only -- login, one candles request, logout.

Requires the PythonGetaway gateway running at trader.config's gateway.base_url
(default http://127.0.0.1:8000) and PYGW_LOGIN/PYGW_PASSWORD/PYGW_SERVER set
(via trader/.env or the environment).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

SYMBOL = "EURUSD"
TIMEFRAME = "H1"
COUNT = 20


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)

    session = client.login()
    print(f"Conectado: slot={session.slot_id}, servidor={config.gateway.server}")

    try:
        df = client.candles(SYMBOL, TIMEFRAME, COUNT)
        print(f"\nUltimas {len(df)} velas {TIMEFRAME} de {SYMBOL} (timestamp en UTC real, corregido de hora de servidor):\n")
        print(df.to_string(index=False))
    finally:
        client.logout()
        print("\nSesion cerrada.")


if __name__ == "__main__":
    main()
