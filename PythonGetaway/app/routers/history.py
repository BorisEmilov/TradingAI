"""Historico: ordenes ejecutadas/canceladas y deals (operaciones reales)."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
from fastapi import APIRouter, Query

from app import converters, mt5_session
from app.errors import BadRequestError

router = APIRouter(prefix="/history", tags=["history"])

_DEFAULT_DAYS = 30


def _dt(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)


def _range(from_epoch: int | None, to_epoch: int | None) -> tuple[datetime, datetime, int, int]:
    now = int(time.time())
    to_e = to_epoch if to_epoch is not None else now + 86400  # margen por desfase de servidor
    from_e = from_epoch if from_epoch is not None else to_e - _DEFAULT_DAYS * 86400
    if to_e <= from_e:
        raise BadRequestError("`to_epoch` debe ser mayor que `from_epoch`.")
    return _dt(from_e), _dt(to_e), from_e, to_e


@router.get("/orders", summary="Ordenes historicas (ejecutadas, canceladas, expiradas)")
def history_orders(
    from_epoch: int | None = Query(None, description="Epoch s. Por defecto: hace 30 dias."),
    to_epoch: int | None = Query(None, description="Epoch s. Por defecto: ahora + 1 dia."),
    symbol: str | None = Query(None),
    group: str | None = Query(None, description=r"Filtro con comodines, ej. `*USD*`"),
    ticket: int | None = Query(None, description="Una orden concreta (ignora el rango de fechas)."),
    position: int | None = Query(None, description="Todas las ordenes de una posicion (ignora el rango)."),
) -> dict:
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        if ticket is not None:
            raw = mt5.history_orders_get(ticket=ticket)
        elif position is not None:
            raw = mt5.history_orders_get(position=position)
        else:
            dfrom, dto, _, _ = _range(from_epoch, to_epoch)
            if symbol:
                raw = mt5.history_orders_get(dfrom, dto, group=symbol)
            elif group:
                raw = mt5.history_orders_get(dfrom, dto, group=group)
            else:
                raw = mt5.history_orders_get(dfrom, dto)
        raw = raw or ()
        items = [converters.order_to_dict(o) for o in raw]
    return {"count": len(items), "orders": items}


@router.get("/deals", summary="Deals historicos (las ejecuciones reales)")
def history_deals(
    from_epoch: int | None = Query(None, description="Epoch s. Por defecto: hace 30 dias."),
    to_epoch: int | None = Query(None, description="Epoch s. Por defecto: ahora + 1 dia."),
    symbol: str | None = Query(None),
    group: str | None = Query(None),
    ticket: int | None = Query(None),
    position: int | None = Query(None, description="Todos los deals de una posicion: sirve para reconstruir su P&L real."),
) -> dict:
    """Un deal es un hecho consumado (entrada o salida). El P&L realizado de una
    posicion es la suma de `profit + swap + commission + fee` de sus deals.

    Los movimientos de saldo (deposito, retirada) tambien aparecen aqui, con
    `type_name = "BALANCE"`."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        if ticket is not None:
            raw = mt5.history_deals_get(ticket=ticket)
        elif position is not None:
            raw = mt5.history_deals_get(position=position)
        else:
            dfrom, dto, _, _ = _range(from_epoch, to_epoch)
            if symbol:
                raw = mt5.history_deals_get(dfrom, dto, group=symbol)
            elif group:
                raw = mt5.history_deals_get(dfrom, dto, group=group)
            else:
                raw = mt5.history_deals_get(dfrom, dto)
        raw = raw or ()
        items = [converters.deal_to_dict(d) for d in raw]
    net = sum(d.get("profit", 0.0) + d.get("swap", 0.0) + d.get("commission", 0.0)
              + d.get("fee", 0.0) for d in items)
    return {"count": len(items), "net_result": round(net, 2), "deals": items}


@router.get("/deals/total", summary="Numero de deals en un rango")
def deals_total(from_epoch: int | None = Query(None), to_epoch: int | None = Query(None)) -> dict:
    mt5_session.require_connection()
    dfrom, dto, _, _ = _range(from_epoch, to_epoch)
    with mt5_session.MT5_LOCK:
        return {"total": mt5.history_deals_total(dfrom, dto)}


@router.get("/orders/total", summary="Numero de ordenes historicas en un rango")
def orders_total(from_epoch: int | None = Query(None), to_epoch: int | None = Query(None)) -> dict:
    mt5_session.require_connection()
    dfrom, dto, _, _ = _range(from_epoch, to_epoch)
    with mt5_session.MT5_LOCK:
        return {"total": mt5.history_orders_total(dfrom, dto)}


@router.get("/position/{position_id}", summary="Vida completa de una posicion cerrada")
def position_history(position_id: int) -> dict:
    """Reune las ordenes y los deals de una posicion y calcula su P&L realizado.

    Es la forma comoda de auditar "que paso exactamente con el ticket X"."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        orders = mt5.history_orders_get(position=position_id) or ()
        deals = mt5.history_deals_get(position=position_id) or ()
        orders_out = [converters.order_to_dict(o) for o in orders]
        deals_out = [converters.deal_to_dict(d) for d in deals]
    realized = sum(d.get("profit", 0.0) + d.get("swap", 0.0) + d.get("commission", 0.0)
                   + d.get("fee", 0.0) for d in deals_out)
    entries = [d for d in deals_out if d.get("entry_name") == "IN"]
    exits = [d for d in deals_out if d.get("entry_name") in ("OUT", "OUT_BY")]
    return {
        "position_id": position_id,
        "found": bool(orders_out or deals_out),
        "realized_pnl": round(realized, 2),
        "volume_in": round(sum(d.get("volume", 0.0) for d in entries), 8),
        "volume_out": round(sum(d.get("volume", 0.0) for d in exits), 8),
        "orders": orders_out,
        "deals": deals_out,
    }
