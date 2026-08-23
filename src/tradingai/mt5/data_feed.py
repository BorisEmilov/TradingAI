"""Espera a que cierre una nueva vela antes de disparar un ciclo del pipeline.

Evita re-procesar la misma vela en curso: solo notifica cuando el timestamp
de la ultima vela cerrada cambia respecto a la anterior observacion.
"""

from __future__ import annotations

import time

from loguru import logger

from tradingai.mt5.connector import MT5Connector


class CandleCloseWatcher:
    def __init__(self, connector: MT5Connector, symbol: str, timeframe: str, poll_seconds: int = 5) -> None:
        self.connector = connector
        self.symbol = symbol
        self.timeframe = timeframe
        self.poll_seconds = poll_seconds
        self._last_timestamp = None

    def wait_for_new_candle(self):
        """Bloquea hasta que hay una vela cerrada nueva; la devuelve como fila de DataFrame."""
        while True:
            candles = self.connector.get_candles(self.symbol, self.timeframe, n_candles=2)
            last_closed = candles.iloc[-2]  # la ultima fila suele ser la vela en formacion

            if self._last_timestamp is None or last_closed["timestamp"] > self._last_timestamp:
                self._last_timestamp = last_closed["timestamp"]
                logger.debug(f"[{self.symbol}] Nueva vela cerrada: {last_closed['timestamp']}")
                return last_closed

            time.sleep(self.poll_seconds)
