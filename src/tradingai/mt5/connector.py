"""Cliente HTTP del bridge MT5 (ver wine_bridge/server.py).

En Linux no se puede importar el paquete `MetaTrader5` directamente: envuelve la DLL
del terminal MT5, que es una aplicacion Windows. En su lugar, un servidor ligero
(`wine_bridge/server.py`) corre dentro de un prefijo Wine junto al terminal y expone
velas, info de cuenta y envio de ordenes por HTTP en localhost. Este conector es el
cliente de esa API; no sabe nada de Wine ni del paquete MetaTrader5.
"""

from __future__ import annotations

import pandas as pd
import requests
from loguru import logger


class MT5ConnectionError(RuntimeError):
    pass


class MT5Connector:
    def __init__(self, base_url: str = "http://127.0.0.1:18812", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._connected = False

    def connect(self) -> None:
        try:
            health = self._get("/health")
        except requests.RequestException as exc:
            raise MT5ConnectionError(
                f"No se pudo contactar el bridge MT5 en {self.base_url}. "
                f"¿Esta corriendo scripts/start_mt5_bridge.sh? Detalle: {exc}"
            ) from exc

        if not health.get("mt5_initialized"):
            raise MT5ConnectionError("El bridge respondio pero MT5 no esta inicializado en el terminal.")

        self._connected = True
        logger.info(f"Conectado al bridge MT5 en {self.base_url}")

    def disconnect(self) -> None:
        self._connected = False

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Conector no inicializado. Llama a connect() primero.")

    def _get(self, path: str, **params) -> dict:
        resp = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_candles(self, symbol: str, timeframe: str, n_candles: int = 500) -> pd.DataFrame:
        self._require_connected()
        data = self._get("/candles", symbol=symbol, timeframe=timeframe, n=n_candles)
        if "error" in data:
            raise RuntimeError(f"No se pudieron obtener velas para {symbol}: {data['error']}")

        df = pd.DataFrame(data["candles"])
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_symbol_info(self, symbol: str) -> dict:
        self._require_connected()
        info = self._get("/symbol_info", symbol=symbol)
        if info is None:
            raise ValueError(f"Simbolo no encontrado: {symbol}")
        return info

    def get_symbol_tick(self, symbol: str) -> dict:
        self._require_connected()
        tick = self._get("/symbol_tick", symbol=symbol)
        if tick is None:
            raise ValueError(f"No hay tick disponible para: {symbol}")
        return tick

    def get_account_info(self) -> dict:
        self._require_connected()
        info = self._get("/account")
        if info is None:
            raise RuntimeError("No se pudo obtener info de cuenta desde el bridge.")
        return info

    def get_open_positions_count(self) -> int:
        self._require_connected()
        return self._get("/positions/count")["count"]

    def get_open_positions(self) -> list[dict]:
        """Cada posicion: {"symbol": str, "type": "buy"|"sell", "volume": float, "profit": float}."""
        self._require_connected()
        return self._get("/positions")["positions"]

    def send_order(self, order_request: dict) -> dict:
        self._require_connected()
        return self._post("/order", order_request)
