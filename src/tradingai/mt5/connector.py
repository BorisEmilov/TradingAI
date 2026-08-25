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

    def calc_profit(self, symbol: str, order_type: str, volume: float, price_open: float, price_close: float) -> float:
        """Profit/perdida exacto que MT5 calcularia para ese movimiento de precio.

        Evita reconstruir a mano la conversion $/lote de cada instrumento (que difiere
        entre forex, CFD e indices -- ver el bug real de XAUUSD del 2026-08-24, donde
        `trade_tick_value/trade_tick_size` no representaba el $ real por punto para un
        simbolo en modo CFD, dando lotes 10x mas grandes de lo debido).
        """
        self._require_connected()
        data = self._get(
            "/order_calc_profit", symbol=symbol, type=order_type, volume=volume,
            price_open=price_open, price_close=price_close,
        )
        if "error" in data:
            raise RuntimeError(f"No se pudo calcular el profit para {symbol}: {data['error']}")
        return data["profit"]

    def calc_margin(self, symbol: str, order_type: str, volume: float, price: float) -> float:
        """Margen exacto que MT5 exigiria para esta orden (respeta convenciones propias
        del instrumento: contract size, leverage, tipo de margen forex vs CFD/indice)."""
        self._require_connected()
        data = self._get("/order_calc_margin", symbol=symbol, type=order_type, volume=volume, price=price)
        if "error" in data:
            raise RuntimeError(f"No se pudo calcular el margen para {symbol}: {data['error']}")
        return data["margin"]

    def get_open_positions_count(self) -> int:
        self._require_connected()
        return self._get("/positions/count")["count"]

    def get_open_positions(self) -> list[dict]:
        """Cada posicion: {"ticket": int, "symbol": str, "type": "buy"|"sell", "volume": float,
        "profit": float, "price_open": float, "sl": float, "tp": float}."""
        self._require_connected()
        return self._get("/positions")["positions"]

    def send_order(self, order_request: dict) -> dict:
        self._require_connected()
        return self._post("/order", order_request)

    def get_position_history(self, ticket: int) -> list[dict]:
        """Deals asociados a una posicion (apertura + cierre), para reportar el
        resultado real cuando el broker cierra por SL/TP sin que el codigo lo pida."""
        self._require_connected()
        return self._get("/history_deals", ticket=ticket)["deals"]

    def modify_position_sl(self, ticket: int, sl: float, tp: float | None = None) -> dict:
        """Mueve el SL (y opcionalmente el TP) de una posicion abierta sin cerrarla.

        Usado por el trailing stop basado en estructura (ver mt5.trailing_stop): a
        diferencia de `send_order`/`close_position`, no abre ni cierra nada, solo
        actualiza los niveles de una posicion existente.
        """
        self._require_connected()
        payload: dict = {"ticket": ticket, "sl": sl}
        if tp is not None:
            payload["tp"] = tp
        return self._post("/positions/modify", payload)

    def close_position(self, ticket: int, volume: float | None = None) -> dict:
        """Cierra una posicion abierta por su ticket (ver `get_open_positions`).

        `volume=None` cierra la posicion completa; un valor menor hace un cierre
        parcial. La cuenta opera en modo hedging, asi que una orden opuesta simple NO
        cierra la posicion (abriria una nueva en sentido contrario) -- el bridge
        referencia el ticket explicitamente via el campo `position` de MT5.
        """
        self._require_connected()
        payload = {"ticket": ticket}
        if volume is not None:
            payload["volume"] = volume
        return self._post("/positions/close", payload)
