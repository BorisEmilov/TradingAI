"""HTTP client for the PythonGetaway MT5 gateway (../PythonGetaway).

Only wraps what this agent needs (auth, multi-timeframe candles, positions,
trading). Placing/closing/modifying orders is exposed but never invoked
automatically by this module -- callers decide when to act on a signal.

IMPORTANT (broker server time): the gateway deliberately does NOT convert MT5
candle/tick timestamps to UTC -- `time` is an epoch computed from the
broker's server wall clock (commonly UTC+2/+3, EET-like), not real UTC (see
PythonGetaway/app/converters.py). Since this agent's session classification
(sessions.py) is DST-aware and needs true UTC, `server_utc_offset()` measures
the live offset from the most recent tick and every candle timestamp is
corrected by it before reaching the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from trader.config import GatewayConfig
from trader.events import TIMEFRAMES

_OFFSET_CACHE_TTL = pd.Timedelta(minutes=30)
_OFFSET_MAX_RESIDUAL_S = 30  # tick fresco de FX líquido: la medición cae a segundos de un múltiplo de 15 min
_TRANSIENT_STATUS_CODES = {502, 503, 504}
_TRANSIENT_RETRIES = 2
_TRANSIENT_BACKOFF_SECONDS = 2.0


def _request_with_retry(method: str, url: str, **kwargs: Any) -> requests.Response:
    """A handful of gateway calls have failed with a one-off 502 (worker
    hiccup) that succeeded immediately on a plain retry -- seen twice across
    unrelated symbols/runs, never reproducible, never explained by the data
    itself (the same request succeeds seconds later). Retrying a bounded
    number of times on the standard transient HTTP statuses is cheap and
    avoids losing a multi-minute fetch to a blip that isn't really an error.
    """
    resp = None
    for attempt in range(_TRANSIENT_RETRIES + 1):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code not in _TRANSIENT_STATUS_CODES:
            return resp
        if attempt < _TRANSIENT_RETRIES:
            time.sleep(_TRANSIENT_BACKOFF_SECONDS)
    return resp  # exhausted retries, still transient -- let raise_for_status() report it normally


class GatewayError(RuntimeError):
    pass


@dataclass
class GatewaySession:
    token: str
    slot_id: str


class PythonGetawayClient:
    def __init__(self, config: GatewayConfig):
        self._config = config
        self._session: GatewaySession | None = None
        self._offset_cache: tuple[pd.Timestamp, pd.Timedelta] | None = None
        self._last_good_offset: pd.Timedelta | None = None

    # -- session -----------------------------------------------------------

    def login(self) -> GatewaySession:
        if not (self._config.login and self._config.password and self._config.server):
            raise GatewayError("missing PYGW_LOGIN/PYGW_PASSWORD/PYGW_SERVER in environment")
        data = self._raw_request(
            "POST",
            "/auth/login",
            timeout=self._config.login_timeout_seconds,
            json={"login": self._config.login, "password": self._config.password, "server": self._config.server},
        )
        self._session = GatewaySession(token=data["token"], slot_id=data["slot_id"])
        return self._session

    def refresh(self) -> None:
        self._request("POST", "/auth/refresh")

    def logout(self) -> None:
        if self._session is not None:
            self._request("POST", "/auth/logout")
            self._session = None

    def _headers(self) -> dict[str, str]:
        if self._session is None:
            raise GatewayError("not logged in")
        return {"Authorization": f"Bearer {self._session.token}"}

    def _raw_request(self, method: str, path: str, timeout: int | None = None, **kwargs: Any) -> dict:
        resp = _request_with_retry(
            method, f"{self._config.base_url}{path}", timeout=timeout or self._config.timeout_seconds, **kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def _request(self, method: str, path: str, timeout: int | None = None, **kwargs: Any) -> dict:
        resp = _request_with_retry(
            method,
            f"{self._config.base_url}{path}",
            headers=self._headers(),
            timeout=timeout or self._config.timeout_seconds,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    # -- time correction -----------------------------------------------------

    def server_utc_offset(self, probe_symbol: str = "EURUSD") -> pd.Timedelta:
        """Broker-server-clock minus real UTC, measured from a live tick.

        Cached for `_OFFSET_CACHE_TTL` -- cheap to recompute but no need to hit
        the gateway on every single candle fetch.

        Solo vale con un tick FRESCO: con mercado cerrado el último tick es
        viejo y (tick - ahora) mezcla offset + antigüedad. Bug real 2026-09-19:
        CSVs bajados un sábado quedaron corridos +23h (velas los sábados, nada
        los domingos). Un offset real es múltiplo de 15 min y está en
        [-12h, +14h]; si la medición no cumple eso, se usa el último offset
        válido, y si nunca hubo uno se falla en voz alta.
        """
        now = pd.Timestamp.now(tz="UTC")
        if self._offset_cache is not None:
            measured_at, offset = self._offset_cache
            if now - measured_at < _OFFSET_CACHE_TTL:
                return offset

        local_now = pd.Timestamp.now(tz="UTC")
        tick = self._request("GET", f"/market/tick/{probe_symbol}")
        server_ts = pd.Timestamp.fromtimestamp(tick["time"], tz="UTC")
        raw_offset_seconds = (server_ts - local_now).total_seconds()
        rounded_seconds = round(raw_offset_seconds / 900.0) * 900
        fresh = abs(raw_offset_seconds - rounded_seconds) <= _OFFSET_MAX_RESIDUAL_S and \
            -12 * 3600 <= rounded_seconds <= 14 * 3600
        if fresh:
            offset = pd.Timedelta(seconds=rounded_seconds)
            self._last_good_offset = offset
        elif self._last_good_offset is not None:
            offset = self._last_good_offset  # tick viejo (mercado cerrado): conservar el último válido
        else:
            raise RuntimeError(
                f"offset de servidor no determinable: último tick de {probe_symbol} viejo "
                f"(medido {raw_offset_seconds / 3600:+.2f}h) -- ¿mercado cerrado? No se bajan datos con fechas corridas."
            )
        self._offset_cache = (now, offset)
        return offset

    # -- market data -----------------------------------------------------

    def candles(self, symbol: str, timeframe: str, count: int, timeout: int | None = None) -> pd.DataFrame:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"unknown timeframe: {timeframe}")
        data = self._request(
            "GET",
            f"/market/candles/{symbol}",
            params={"timeframe": timeframe, "count": count, "include_current": False},
            timeout=timeout,
        )
        rows = data["candles"]
        offset = self.server_utc_offset()

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True) - offset
        df = df.rename(columns={"tick_volume": "volume"})
        keep = ["timestamp", "open", "high", "low", "close"]
        keep += ["volume"] if "volume" in df.columns else []
        keep += ["spread"] if "spread" in df.columns else []
        df = df[keep].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df

    def fetch_multi_timeframe(self, symbol: str, counts: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
        counts = counts or {"D1": 500, "H1": 1000, "M15": 2000}
        return {tf: self.candles(symbol, tf, counts.get(tf, 500)) for tf in ("D1", "H1", "M15")}

    def last_price(self, symbol: str) -> float:
        bid, ask = self.bid_ask(symbol)
        return (bid + ask) / 2.0

    def bid_ask(self, symbol: str) -> tuple[float, float]:
        tick = self._request("GET", f"/market/tick/{symbol}")
        return float(tick["bid"]), float(tick["ask"])

    def recent_avg_spread(self, symbol: str, minutes: int = 15) -> float:
        """Spread bid-ask PROMEDIO (en precio) de los ticks reales de los últimos
        `minutes`. NO usar el campo `spread` de las velas para costo: MT5 guarda
        ahí el spread MÍNIMO de la vela (verificado 2026-09-25 contra ticks,
        320/320 velas) -- desde jul-2026 ese mínimo es 0 casi siempre aunque el
        spread real no lo sea. Sin ticks en la ventana -> bid-ask del último tick."""
        last = self._request("GET", f"/market/tick/{symbol}")
        to_epoch = int(last["time"]) + 1  # epoch de servidor, mismo reloj que espera /ticks/range
        data = self._request(
            "GET", f"/market/ticks/{symbol}/range", timeout=60,
            params={"from_epoch": to_epoch - minutes * 60, "to_epoch": to_epoch, "flags": "INFO"},
        )
        spreads = [t["ask"] - t["bid"] for t in data["ticks"] if t["bid"] > 0 and t["ask"] > 0]
        return sum(spreads) / len(spreads) if spreads else float(last["ask"] - last["bid"])

    def symbol_info(self, symbol: str, timeout: int | None = None) -> dict:
        return self._request("GET", f"/symbols/{symbol}", timeout=timeout)

    def list_symbols(self, group: str | None = None, only_visible: bool = False, timeout: int | None = None) -> list[str]:
        """Symbol names only (fast) -- the broker exposes 12000+ symbols total,
        so `names_only=false` (full spec for every one) is deliberately never
        used here; fetch specs one at a time for a shortlist instead."""
        params: dict[str, Any] = {"only_visible": only_visible, "names_only": True}
        if group:
            params["group"] = group
        data = self._request("GET", "/symbols", params=params, timeout=timeout)
        return data["symbols"]

    # -- trading -----------------------------------------------------

    def open_position(self, symbol: str, direction: str, volume: float, sl: float, tp: float, comment: str = "") -> dict:
        side = "BUY" if direction == "long" else "SELL"
        return self._request(
            "POST",
            "/trading/open",
            json={"symbol": symbol, "side": side, "volume": volume, "sl": sl, "tp": tp, "comment": comment[:31]},
        )

    def close_position(self, ticket: int) -> dict:
        return self._request("POST", f"/trading/close/{ticket}", json={})

    def close_partial(self, ticket: int, volume: float) -> dict:
        return self._request("POST", f"/trading/close-partial/{ticket}", json={"volume": volume})

    def modify_sl(self, ticket: int, sl: float) -> dict:
        return self._request("POST", f"/trading/modify-sl/{ticket}", json={"sl": sl})

    def place_pending(self, symbol: str, order_type: str, volume: float, price: float,
                       sl: float, tp: float, expiration: int, comment: str = "") -> dict:
        """`order_type`: "BUY_LIMIT" | "SELL_LIMIT". `expiration`: epoch en
        segundos EN HORA DE SERVIDOR (confirmado en PythonGetaway/app/models.py
        `PlacePendingRequest.expiration` -- misma convención de broker-server-time
        que el resto de este cliente, nunca UTC real sin corregir)."""
        return self._request(
            "POST",
            "/trading/pending",
            json={"symbol": symbol, "type": order_type, "volume": volume, "price": price,
                  "sl": sl, "tp": tp, "expiration": expiration, "comment": comment[:31]},
        )

    def cancel_pending(self, ticket: int) -> dict:
        return self._request("DELETE", f"/trading/pending/{ticket}")

    def pending_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else None
        data = self._request("GET", "/orders", params=params)
        return data if isinstance(data, list) else data.get("orders", [])

    def order_history(self, ticket: int) -> dict:
        """Historial de una orden (pendiente) ya no viva -- estado final real
        (FILLED/CANCELED/EXPIRED/REJECTED), para loguear el motivo real por el
        que una orden límite dejó de estar en `/orders` en vez de asumirlo."""
        return self._request("GET", "/history/orders", params={"ticket": ticket})

    def positions(self) -> list[dict]:
        data = self._request("GET", "/positions")
        return data if isinstance(data, list) else data.get("positions", [])

    def account(self) -> dict:
        """Balance, equity, margin, trade_mode (0=DEMO/1=CONTEST/2=REAL)..."""
        return self._request("GET", "/account")

    def position_history(self, ticket: int) -> dict:
        """Vida completa de una posición (abierta o ya cerrada): sus deals
        (entradas/salidas reales, con precio y motivo -- SL/TP/CLIENT/...) y
        el P&L realizado. Necesario para reconstruir qué pasó con una posición
        que el bot no cerró él mismo (SL o TP real ejecutados por el broker)."""
        return self._request("GET", f"/history/position/{ticket}")
