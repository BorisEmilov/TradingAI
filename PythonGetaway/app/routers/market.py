"""Datos de mercado: cotizaciones, velas, ticks y profundidad de mercado."""

from __future__ import annotations

from datetime import datetime, timezone

import MetaTrader5 as mt5
from fastapi import APIRouter, Query

from app import config, converters, mt5_session
from app.errors import BadRequestError, MT5Error, NotFoundError

router = APIRouter(prefix="/market", tags=["market"])

_TF_HELP = "Timeframe: " + ", ".join(converters.TIMEFRAMES)


def _tf(timeframe: str) -> int:
    key = timeframe.upper()
    if key not in converters.TIMEFRAMES:
        raise BadRequestError(f"Timeframe invalido: {timeframe}",
                              detail={"valid": sorted(converters.TIMEFRAMES)})
    return converters.TIMEFRAMES[key]


def _dt(epoch: int) -> datetime:
    """MT5 espera `datetime`. Se construye en UTC para que el epoch que llega
    del cliente se use tal cual, sin que la zona local del servidor lo desplace."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)


@router.get("/tick/{symbol}", summary="Ultimo tick (cotizacion actual)")
def last_tick(symbol: str) -> dict:
    """bid, ask, last, volume, spread calculado y hora del tick."""
    mt5_session.require_connection()
    info = mt5_session.ensure_symbol(symbol)
    with mt5_session.MT5_LOCK:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise NotFoundError(f"Sin cotizacion disponible para {symbol}.",
                                detail=mt5_session.last_error())
        data = converters.tick_to_dict(tick)
    data["symbol"] = symbol
    data["digits"] = info.digits
    data["point"] = info.point
    if data.get("ask") and data.get("bid"):
        data["spread_price"] = round(data["ask"] - data["bid"], info.digits)
        data["spread_points"] = round((data["ask"] - data["bid"]) / info.point) if info.point else None
    return data


@router.get("/ticks", summary="Ultimo tick de varios simbolos a la vez")
def last_ticks(symbols: str = Query(..., description="Nombres separados por coma. Ej: `EURUSD,GBPUSD,USDJPY`")) -> dict:
    """Una sola llamada en lugar de N. Los simbolos que fallen se reportan en
    `errors` en vez de tumbar la peticion entera."""
    mt5_session.require_connection()
    names = [s.strip() for s in symbols.split(",") if s.strip()]
    if not names:
        raise BadRequestError("Parametro `symbols` vacio.")
    out, errors = {}, {}
    for name in names:
        try:
            out[name] = last_tick(name)
        except Exception as exc:  # noqa: BLE001 -- un simbolo malo no debe romper el lote
            errors[name] = getattr(exc, "detail", {"error": str(exc)})
    return {"count": len(out), "ticks": out, "errors": errors}


@router.get("/candles/{symbol}", summary="Velas OHLCV por posicion (las N mas recientes)")
def candles_from_pos(
    symbol: str,
    timeframe: str = Query("M15", description=_TF_HELP),
    count: int = Query(500, ge=1, description="Numero de velas."),
    start_pos: int = Query(0, ge=0, description="0 = la vela EN CURSO (aun sin cerrar). 1 = la ultima vela cerrada."),
    include_current: bool = Query(True, description="false = descarta la vela en formacion. Util para no operar con una vela incompleta."),
) -> dict:
    """La vela de indice 0 es la que se esta formando AHORA y cambia a cada tick.

    Si vas a tomar decisiones sobre velas cerradas, usa `include_current=false`
    (equivale a pedir desde `start_pos=1`)."""
    mt5_session.require_connection()
    if count > config.MAX_BARS:
        raise BadRequestError(f"count maximo permitido: {config.MAX_BARS}", detail={"count": count})
    mt5_session.ensure_symbol(symbol)
    pos = start_pos + (0 if include_current else 1)
    with mt5_session.MT5_LOCK:
        rates = mt5.copy_rates_from_pos(symbol, _tf(timeframe), pos, count)
        if rates is None:
            raise MT5Error(f"copy_rates_from_pos fallo para {symbol}.", **mt5_session.last_error())
    return {"symbol": symbol, "timeframe": timeframe.upper(), "count": len(rates),
            "candles": converters.rates_to_list(rates)}


@router.get("/candles/{symbol}/from", summary="Velas OHLCV desde una fecha")
def candles_from(
    symbol: str,
    timeframe: str = Query("M15", description=_TF_HELP),
    from_epoch: int = Query(..., description="Epoch en segundos (hora de servidor)."),
    count: int = Query(500, ge=1),
) -> dict:
    mt5_session.require_connection()
    if count > config.MAX_BARS:
        raise BadRequestError(f"count maximo permitido: {config.MAX_BARS}")
    mt5_session.ensure_symbol(symbol)
    with mt5_session.MT5_LOCK:
        rates = mt5.copy_rates_from(symbol, _tf(timeframe), _dt(from_epoch), count)
        if rates is None:
            raise MT5Error(f"copy_rates_from fallo para {symbol}.", **mt5_session.last_error())
    return {"symbol": symbol, "timeframe": timeframe.upper(), "count": len(rates),
            "candles": converters.rates_to_list(rates)}


@router.get("/candles/{symbol}/range", summary="Velas OHLCV entre dos fechas")
def candles_range(
    symbol: str,
    timeframe: str = Query("M15", description=_TF_HELP),
    from_epoch: int = Query(..., description="Epoch en segundos (hora de servidor)."),
    to_epoch: int = Query(..., description="Epoch en segundos (hora de servidor)."),
) -> dict:
    mt5_session.require_connection()
    if to_epoch <= from_epoch:
        raise BadRequestError("`to_epoch` debe ser mayor que `from_epoch`.")
    mt5_session.ensure_symbol(symbol)
    with mt5_session.MT5_LOCK:
        rates = mt5.copy_rates_range(symbol, _tf(timeframe), _dt(from_epoch), _dt(to_epoch))
        if rates is None:
            raise MT5Error(f"copy_rates_range fallo para {symbol}.", **mt5_session.last_error())
    return {"symbol": symbol, "timeframe": timeframe.upper(), "count": len(rates),
            "candles": converters.rates_to_list(rates)}


@router.get("/ticks/{symbol}/from", summary="Ticks crudos desde una fecha")
def ticks_from(
    symbol: str,
    from_epoch: int = Query(..., description="Epoch en segundos (hora de servidor)."),
    count: int = Query(1000, ge=1),
    flags: str = Query("ALL", description="ALL | INFO | TRADE. ALL devuelve todos los ticks."),
) -> dict:
    mt5_session.require_connection()
    if count > config.MAX_TICKS:
        raise BadRequestError(f"count maximo permitido: {config.MAX_TICKS}")
    mt5_session.ensure_symbol(symbol)
    flag_map = {"ALL": mt5.COPY_TICKS_ALL, "INFO": mt5.COPY_TICKS_INFO, "TRADE": mt5.COPY_TICKS_TRADE}
    if flags.upper() not in flag_map:
        raise BadRequestError("flags debe ser ALL, INFO o TRADE.")
    with mt5_session.MT5_LOCK:
        ticks = mt5.copy_ticks_from(symbol, _dt(from_epoch), count, flag_map[flags.upper()])
        if ticks is None:
            raise MT5Error(f"copy_ticks_from fallo para {symbol}.", **mt5_session.last_error())
    return {"symbol": symbol, "count": len(ticks), "ticks": converters.ticks_to_list(ticks)}


@router.get("/ticks/{symbol}/range", summary="Ticks crudos entre dos fechas")
def ticks_range(
    symbol: str,
    from_epoch: int = Query(...),
    to_epoch: int = Query(...),
    flags: str = Query("ALL", description="ALL | INFO | TRADE"),
) -> dict:
    mt5_session.require_connection()
    if to_epoch <= from_epoch:
        raise BadRequestError("`to_epoch` debe ser mayor que `from_epoch`.")
    mt5_session.ensure_symbol(symbol)
    flag_map = {"ALL": mt5.COPY_TICKS_ALL, "INFO": mt5.COPY_TICKS_INFO, "TRADE": mt5.COPY_TICKS_TRADE}
    if flags.upper() not in flag_map:
        raise BadRequestError("flags debe ser ALL, INFO o TRADE.")
    with mt5_session.MT5_LOCK:
        ticks = mt5.copy_ticks_range(symbol, _dt(from_epoch), _dt(to_epoch), flag_map[flags.upper()])
        if ticks is None:
            raise MT5Error(f"copy_ticks_range fallo para {symbol}.", **mt5_session.last_error())
    return {"symbol": symbol, "count": len(ticks), "ticks": converters.ticks_to_list(ticks)}


@router.get("/book/{symbol}", summary="Profundidad de mercado (DOM / Level II)")
def market_book(symbol: str) -> dict:
    """Suscribe el simbolo al libro, lee y devuelve los niveles.

    Muchos brokers de Forex retail NO publican profundidad: en ese caso
    `entries` viene vacio y `subscribed` indica si la suscripcion se acepto.
    No es un error del gateway."""
    mt5_session.require_connection()
    mt5_session.ensure_symbol(symbol)
    with mt5_session.MT5_LOCK:
        subscribed = mt5.market_book_add(symbol)
        entries = mt5.market_book_get(symbol) if subscribed else None
        result = [converters.book_to_dict(e) for e in entries] if entries else []
        if subscribed:
            mt5.market_book_release(symbol)
    return {"symbol": symbol, "subscribed": bool(subscribed), "count": len(result),
            "entries": result,
            "note": None if result else "El broker no publica profundidad de mercado para este simbolo."}
