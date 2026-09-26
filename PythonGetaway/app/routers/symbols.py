"""Catalogo de simbolos y sus especificaciones."""

from __future__ import annotations

import MetaTrader5 as mt5
from fastapi import APIRouter, Query

from app import converters, mt5_session
from app.errors import MT5Error, NotFoundError

router = APIRouter(prefix="/symbols", tags=["symbols"])


@router.get("", summary="Lista de simbolos del broker")
def list_symbols(
    group: str | None = Query(None, description=r"Filtro estilo MT5, admite comodines. Ej: `*USD*`, `*,!*EUR*` (todos menos los que lleven EUR)."),
    only_visible: bool = Query(False, description="true = solo los que estan en Market Watch."),
    names_only: bool = Query(True, description="true = solo nombres (rapido). false = especificacion completa de cada simbolo (pesado: son miles)."),
) -> dict:
    """Con `names_only=false` y sin `group` esto devuelve MUCHOS megas -- el
    broker de prueba expone >12.000 simbolos. Filtra siempre que puedas."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        symbols = mt5.symbols_get(group) if group else mt5.symbols_get()
        if symbols is None:
            raise MT5Error("symbols_get() devolvio None.", **mt5_session.last_error())
        if only_visible:
            symbols = [s for s in symbols if s.visible]
        if names_only:
            return {"count": len(symbols), "symbols": [s.name for s in symbols]}
        return {"count": len(symbols), "symbols": [converters.symbol_to_dict(s) for s in symbols]}


@router.get("/total", summary="Numero total de simbolos")
def symbols_total() -> dict:
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        return {"total": mt5.symbols_total()}


@router.get("/{symbol}", summary="Especificacion completa de un simbolo")
def symbol_info(symbol: str) -> dict:
    """Digitos, punto, spread, volumenes min/max/step, margen, swaps, sesion,
    modo de llenado permitido, divisas base/beneficio... todo lo que MT5 sabe.

    Selecciona el simbolo en Market Watch si aun no lo estaba."""
    mt5_session.require_connection()
    info = mt5_session.ensure_symbol(symbol)
    data = converters.symbol_to_dict(info)
    mask = int(getattr(info, "filling_mode", 0) or 0)
    data["filling_modes_allowed"] = [n for bit, n in ((1, "FOK"), (2, "IOC")) if mask & bit] or ["RETURN"]
    return data


@router.post("/{symbol}/select", summary="Anadir o quitar un simbolo del Market Watch")
def select_symbol(symbol: str, enable: bool = Query(True)) -> dict:
    """Un simbolo que no esta en Market Watch no devuelve cotizaciones ni velas.
    Los endpoints de lectura ya lo seleccionan solos; esto es control manual."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        if mt5.symbol_info(symbol) is None:
            raise NotFoundError(f"Simbolo desconocido para el broker: {symbol}")
        ok = mt5.symbol_select(symbol, enable)
        if not ok:
            raise MT5Error(f"symbol_select({symbol}, {enable}) fallo.", **mt5_session.last_error())
        return {"symbol": symbol, "selected": enable}
