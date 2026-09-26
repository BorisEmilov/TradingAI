"""Conversion de las estructuras de MetaTrader5 a JSON.

SOBRE LOS TIEMPOS (importante para el cliente .NET)
---------------------------------------------------
MT5 devuelve los tiempos como *epoch seconds* medidos en la **hora del servidor
del broker**, no en UTC. Es decir: el numero es un epoch, pero el reloj que lo
produjo es el del broker (MetaQuotes-Demo suele ir en UTC+2/+3).

Este gateway NO inventa una conversion de zona horaria -- seria adivinar. Cada
campo temporal se expone dos veces:

    "time":     1757520000            <- el epoch crudo, tal cual lo da MT5
    "time_iso": "2026-09-10T14:00:00" <- el mismo instante formateado, SIN sufijo Z

La ausencia de `Z`/offset en `time_iso` es deliberada: marca que es hora de
servidor, no UTC. Si necesitas UTC real, usa el offset de tu broker.
En .NET: deserializa `time_iso` como `DateTime` con `DateTimeKind.Unspecified`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import MetaTrader5 as mt5
import numpy as np

# ---------------------------------------------------------------------------
# Mapas de enumeraciones -- se envian junto al valor numerico para que el
# cliente no tenga que replicar las constantes de MQL5.
# ---------------------------------------------------------------------------
ORDER_TYPE_NAMES = {
    0: "BUY", 1: "SELL", 2: "BUY_LIMIT", 3: "SELL_LIMIT",
    4: "BUY_STOP", 5: "SELL_STOP", 6: "BUY_STOP_LIMIT", 7: "SELL_STOP_LIMIT",
    8: "CLOSE_BY",
}
POSITION_TYPE_NAMES = {0: "BUY", 1: "SELL"}
ORDER_STATE_NAMES = {
    0: "STARTED", 1: "PLACED", 2: "CANCELED", 3: "PARTIAL", 4: "FILLED",
    5: "REJECTED", 6: "EXPIRED", 7: "REQUEST_ADD", 8: "REQUEST_MODIFY",
    9: "REQUEST_CANCEL",
}
ORDER_FILLING_NAMES = {0: "FOK", 1: "IOC", 2: "RETURN"}
ORDER_TIME_NAMES = {0: "GTC", 1: "DAY", 2: "SPECIFIED", 3: "SPECIFIED_DAY"}
DEAL_TYPE_NAMES = {
    0: "BUY", 1: "SELL", 2: "BALANCE", 3: "CREDIT", 4: "CHARGE", 5: "CORRECTION",
    6: "BONUS", 7: "COMMISSION", 8: "COMMISSION_DAILY", 9: "COMMISSION_MONTHLY",
    10: "COMMISSION_AGENT_DAILY", 11: "COMMISSION_AGENT_MONTHLY", 12: "INTEREST",
    13: "BUY_CANCELED", 14: "SELL_CANCELED", 15: "DIVIDEND", 16: "DIVIDEND_FRANKED",
    17: "TAX",
}
DEAL_ENTRY_NAMES = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}
DEAL_REASON_NAMES = {
    0: "CLIENT", 1: "MOBILE", 2: "WEB", 3: "EXPERT", 4: "SL", 5: "TP", 6: "SO",
    7: "ROLLOVER", 8: "VMARGIN", 9: "SPLIT",
}
POSITION_REASON_NAMES = {0: "CLIENT", 1: "MOBILE", 2: "WEB", 3: "EXPERT"}
TRADE_MODE_NAMES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}
MARGIN_MODE_NAMES = {0: "RETAIL_NETTING", 1: "EXCHANGE", 2: "RETAIL_HEDGING"}
BOOK_TYPE_NAMES = {1: "SELL", 2: "BUY", 3: "SELL_MARKET", 4: "BUY_MARKET"}

# Retcodes mas frecuentes de order_send/order_check.
RETCODE_NAMES = {
    10004: "REQUOTE", 10006: "REJECT", 10007: "CANCEL", 10008: "PLACED",
    10009: "DONE", 10010: "DONE_PARTIAL", 10011: "ERROR", 10012: "TIMEOUT",
    10013: "INVALID", 10014: "INVALID_VOLUME", 10015: "INVALID_PRICE",
    10016: "INVALID_STOPS", 10017: "TRADE_DISABLED", 10018: "MARKET_CLOSED",
    10019: "NO_MONEY", 10020: "PRICE_CHANGED", 10021: "PRICE_OFF",
    10022: "INVALID_EXPIRATION", 10023: "ORDER_CHANGED", 10024: "TOO_MANY_REQUESTS",
    10025: "NO_CHANGES", 10026: "SERVER_DISABLES_AT", 10027: "CLIENT_DISABLES_AT",
    10028: "LOCKED", 10029: "FROZEN", 10030: "INVALID_FILL", 10031: "CONNECTION",
    10032: "ONLY_REAL", 10033: "LIMIT_ORDERS", 10034: "LIMIT_VOLUME",
    10035: "INVALID_ORDER", 10036: "POSITION_CLOSED", 10038: "INVALID_CLOSE_VOLUME",
    10039: "CLOSE_ORDER_EXIST", 10040: "LIMIT_POSITIONS",
    10041: "REJECT_CANCEL", 10042: "LONG_ONLY", 10043: "SHORT_ONLY",
    10044: "CLOSE_ONLY", 10045: "FIFO_CLOSE", 10046: "HEDGE_PROHIBITED",
}

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4, "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}

# Campos que llevan un epoch en segundos / milisegundos.
_TIME_FIELDS_S = {"time", "time_setup", "time_done", "time_expiration", "time_update",
                  "start_time", "expiration_time", "time_msc_dummy"}
_TIME_FIELDS_MS = {"time_msc", "time_setup_msc", "time_done_msc", "time_update_msc"}


def iso_from_epoch(seconds: float | int | None) -> str | None:
    """Epoch -> ISO-8601 SIN sufijo de zona (es hora de servidor, ver modulo)."""
    if seconds in (None, 0):
        return None
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).replace(tzinfo=None).isoformat()


def _py(value: Any) -> Any:
    """numpy -> tipos nativos de Python (json no serializa np.float64)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def named_to_dict(obj: Any) -> dict:
    """namedtuple de MT5 -> dict, anadiendo `*_iso` a cada campo temporal."""
    if obj is None:
        return {}
    raw = obj._asdict() if hasattr(obj, "_asdict") else dict(obj)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        out[key] = _py(value)
    for key in list(out.keys()):
        if key in _TIME_FIELDS_S and isinstance(out[key], (int, float)):
            out[f"{key}_iso"] = iso_from_epoch(out[key])
        elif key in _TIME_FIELDS_MS and isinstance(out[key], (int, float)):
            out[f"{key}_iso"] = iso_from_epoch(out[key] / 1000.0) if out[key] else None
    return out


def _label(d: dict, field: str, mapping: dict, target: str) -> None:
    if field in d and isinstance(d[field], int):
        d[target] = mapping.get(d[field], "UNKNOWN")


def position_to_dict(p: Any) -> dict:
    d = named_to_dict(p)
    _label(d, "type", POSITION_TYPE_NAMES, "type_name")
    _label(d, "reason", POSITION_REASON_NAMES, "reason_name")
    return d


def order_to_dict(o: Any) -> dict:
    d = named_to_dict(o)
    _label(d, "type", ORDER_TYPE_NAMES, "type_name")
    _label(d, "state", ORDER_STATE_NAMES, "state_name")
    _label(d, "type_filling", ORDER_FILLING_NAMES, "type_filling_name")
    _label(d, "type_time", ORDER_TIME_NAMES, "type_time_name")
    _label(d, "reason", POSITION_REASON_NAMES, "reason_name")
    return d


def deal_to_dict(dl: Any) -> dict:
    d = named_to_dict(dl)
    _label(d, "type", DEAL_TYPE_NAMES, "type_name")
    _label(d, "entry", DEAL_ENTRY_NAMES, "entry_name")
    _label(d, "reason", DEAL_REASON_NAMES, "reason_name")
    return d


def account_to_dict(a: Any) -> dict:
    d = named_to_dict(a)
    _label(d, "trade_mode", TRADE_MODE_NAMES, "trade_mode_name")
    _label(d, "margin_mode", MARGIN_MODE_NAMES, "margin_mode_name")
    return d


def symbol_to_dict(s: Any) -> dict:
    return named_to_dict(s)


def tick_to_dict(t: Any) -> dict:
    return named_to_dict(t)


def book_to_dict(entry: Any) -> dict:
    d = named_to_dict(entry)
    _label(d, "type", BOOK_TYPE_NAMES, "type_name")
    return d


def rates_to_list(rates: Any) -> list[dict]:
    """numpy structured array de copy_rates_* -> lista de dicts."""
    if rates is None:
        return []
    out = []
    for row in rates:
        item = {name: _py(row[name]) for name in rates.dtype.names}
        if "time" in item:
            item["time_iso"] = iso_from_epoch(item["time"])
        out.append(item)
    return out


def ticks_to_list(ticks: Any) -> list[dict]:
    if ticks is None:
        return []
    out = []
    for row in ticks:
        item = {name: _py(row[name]) for name in ticks.dtype.names}
        if "time" in item:
            item["time_iso"] = iso_from_epoch(item["time"])
        if "time_msc" in item and item["time_msc"]:
            item["time_msc_iso"] = iso_from_epoch(item["time_msc"] / 1000.0)
        out.append(item)
    return out


def order_result_to_dict(result: Any, is_check: bool = False) -> dict:
    """OrderSendResult / OrderCheckResult -> dict, con el retcode traducido.

    `is_check=True` cambia la semantica del exito: `order_check` devuelve
    retcode **0** cuando la validacion pasa (no hay un codigo "OK" positivo),
    mientras que `order_send` usa 10008/10009/10010."""
    if result is None:
        return {}
    d = named_to_dict(result)
    if "retcode" in d:
        if is_check:
            d["retcode_name"] = "OK" if d["retcode"] == 0 else RETCODE_NAMES.get(d["retcode"], "UNKNOWN")
            d["success"] = d["retcode"] == 0
        else:
            d["retcode_name"] = RETCODE_NAMES.get(d["retcode"], "UNKNOWN")
            d["success"] = d["retcode"] in (10008, 10009, 10010)
    if "request" in d and hasattr(result, "request"):
        req = named_to_dict(result.request)
        _label(req, "type", ORDER_TYPE_NAMES, "type_name")
        d["request"] = req
    return d
