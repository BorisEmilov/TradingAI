"""Espera a que cierre una nueva vela antes de disparar un ciclo del pipeline.

Evita re-procesar la misma vela en curso: solo notifica cuando el timestamp
de la ultima vela cerrada cambia respecto a la anterior observacion.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from loguru import logger

from tradingai.mt5.connector import MT5Connector


class CandleCloseWatcher:
    def __init__(
        self,
        connector: MT5Connector,
        symbol: str,
        timeframe: str,
        poll_seconds: int = 5,
        state_file: str | Path | None = None,
    ) -> None:
        """`state_file`, si se pasa, persiste la ultima vela vista en disco -- sin
        esto, `_last_timestamp` solo vive en memoria y arranca en None en cada
        reinicio del proceso, asi que la primera llamada trata la vela YA cerrada
        (posiblemente ya evaluada segundos antes del reinicio) como "nueva" y
        dispara un ciclo del pipeline de inmediato en vez de esperar la siguiente
        vela real (caso real 2026-08-28: reiniciar el piloto varias veces seguidas
        para desplegar arreglos disparaba entradas nuevas en cada reinicio)."""
        self.connector = connector
        self.symbol = symbol
        self.timeframe = timeframe
        self.poll_seconds = poll_seconds
        self.state_file = Path(state_file) if state_file else None
        self._last_timestamp = self._load_last_timestamp()

    def _load_last_timestamp(self):
        if self.state_file is None or not self.state_file.exists():
            return None
        try:
            return pd.Timestamp(self.state_file.read_text().strip())
        except Exception:
            logger.exception(f"[{self.symbol}] Error leyendo estado de vela persistido, se ignora")
            return None

    def _save_last_timestamp(self, timestamp) -> None:
        if self.state_file is None:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(str(timestamp))
        except Exception:
            logger.exception(f"[{self.symbol}] Error guardando estado de vela persistido")

    def wait_for_new_candle(self):
        """Bloquea hasta que hay una vela cerrada nueva; la devuelve como fila de DataFrame."""
        while True:
            candles = self.connector.get_candles(self.symbol, self.timeframe, n_candles=2)
            last_closed = candles.iloc[-2]  # la ultima fila suele ser la vela en formacion

            if self._last_timestamp is None or last_closed["timestamp"] > self._last_timestamp:
                self._last_timestamp = last_closed["timestamp"]
                self._save_last_timestamp(self._last_timestamp)
                logger.debug(f"[{self.symbol}] Nueva vela cerrada: {last_closed['timestamp']}")
                return last_closed

            time.sleep(self.poll_seconds)
