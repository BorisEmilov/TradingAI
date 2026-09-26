"""Ordenes pendientes activas (lectura)."""

from __future__ import annotations

import MetaTrader5 as mt5
from fastapi import APIRouter, Query

from app import converters, mt5_session
from app.errors import NotFoundError

router = APIRouter(prefix="/orders", tags=["orders"])


@router.get("", summary="Ordenes pendientes activas")
def list_orders(
    symbol: str | None = Query(None),
    group: str | None = Query(None, description=r"Filtro con comodines. Ej: `*USD*`"),
    magic: int | None = Query(None),
) -> dict:
    """Solo ordenes VIVAS (limit/stop aun no activadas). Las ya ejecutadas o
    canceladas estan en `/history/orders`."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        if symbol:
            raw = mt5.orders_get(symbol=symbol)
        elif group:
            raw = mt5.orders_get(group=group)
        else:
            raw = mt5.orders_get()
        raw = raw or ()
        items = [converters.order_to_dict(o) for o in raw]
    if magic is not None:
        items = [o for o in items if o.get("magic") == magic]
    return {"count": len(items), "orders": items}


@router.get("/total", summary="Numero de ordenes pendientes")
def orders_total() -> dict:
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        return {"total": mt5.orders_total()}


@router.get("/{ticket}", summary="Una orden pendiente por ticket")
def get_order(ticket: int) -> dict:
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        raw = mt5.orders_get(ticket=ticket)
        if not raw:
            raise NotFoundError(f"No hay ninguna orden pendiente con ticket {ticket}.")
        return converters.order_to_dict(raw[0])
