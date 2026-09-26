"""Posiciones abiertas (lectura)."""

from __future__ import annotations

import MetaTrader5 as mt5
from fastapi import APIRouter, Query

from app import converters, mt5_session
from app.errors import NotFoundError

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("", summary="Posiciones abiertas")
def list_positions(
    symbol: str | None = Query(None, description="Filtrar por simbolo exacto."),
    group: str | None = Query(None, description=r"Filtro con comodines. Ej: `*USD*`"),
    magic: int | None = Query(None, description="Filtrar por magic number (se aplica en el gateway)."),
) -> dict:
    """`symbol` y `group` son mutuamente excluyentes: si mandas los dos, manda
    `symbol`. `profit` ya viene neto de swap? No: `profit` es el flotante SIN
    swap ni comision -- esos van en sus propios campos."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        if symbol:
            raw = mt5.positions_get(symbol=symbol)
        elif group:
            raw = mt5.positions_get(group=group)
        else:
            raw = mt5.positions_get()
        raw = raw or ()
        items = [converters.position_to_dict(p) for p in raw]
    if magic is not None:
        items = [p for p in items if p.get("magic") == magic]
    total_profit = sum(p.get("profit", 0.0) for p in items)
    total_volume = sum(p.get("volume", 0.0) for p in items)
    return {"count": len(items), "total_profit": round(total_profit, 2),
            "total_volume": round(total_volume, 8), "positions": items}


@router.get("/total", summary="Numero de posiciones abiertas")
def positions_total() -> dict:
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        return {"total": mt5.positions_total()}


@router.get("/{ticket}", summary="Una posicion por ticket")
def get_position(ticket: int) -> dict:
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        raw = mt5.positions_get(ticket=ticket)
        if not raw:
            raise NotFoundError(f"No hay ninguna posicion abierta con ticket {ticket}.")
        return converters.position_to_dict(raw[0])
