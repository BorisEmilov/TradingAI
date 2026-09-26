"""Trading manual: abrir, cerrar (total o parcial), mover SL/TP y ordenes pendientes.

TODOS los endpoints de este modulo llaman a `require_trading_allowed()`, que
rechaza la operativa si la cuenta conectada es REAL y el gateway no arranco con
`PYGW_ALLOW_REAL=1`.

CONVENCION DE RESPUESTA
-----------------------
Todo lo que envia una orden devuelve la misma forma:

    {
      "success": true,
      "retcode": 10009,
      "retcode_name": "DONE",
      "comment": "Request executed",
      "order": 123456789,      <- ticket de la ORDEN generada
      "deal": 987654321,       <- ticket del DEAL (ejecucion)
      "position": 123456789,   <- ticket de la POSICION resultante (si aplica)
      "volume": 0.1,
      "price": 1.09123,
      "request": { ... }       <- lo que se envio realmente a MT5
    }

`success` es true solo con retcode 10008 (PLACED), 10009 (DONE) o 10010
(DONE_PARTIAL). Un HTTP 200 con `success: false` significa que MT5 acepto la
peticion pero la RECHAZO: mira `retcode_name`.
"""

from __future__ import annotations

import MetaTrader5 as mt5
from fastapi import APIRouter, Path

from app import config, converters, models, mt5_session
from app.errors import BadRequestError, MT5Error, NotFoundError

router = APIRouter(prefix="/trading", tags=["trading"])

_SIDE_TO_ORDER = {"BUY": mt5.ORDER_TYPE_BUY, "SELL": mt5.ORDER_TYPE_SELL}
_PENDING_TO_ORDER = {
    "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT, "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
    "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP, "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
}


def _send(request: dict) -> dict:
    """Envia a MT5 y normaliza la respuesta. Nunca lanza por un retcode malo:
    un rechazo del broker es informacion, no un fallo del gateway."""
    with mt5_session.MT5_LOCK:
        result = mt5.order_send(request)
        if result is None:
            code, desc = mt5.last_error()
            raise MT5Error(f"order_send() devolvio None: {desc}", mt5_code=code,
                           detail={"request": _jsonable(request)})
        out = converters.order_result_to_dict(result)
    out["request"] = _jsonable(request)
    return out


def _jsonable(request: dict) -> dict:
    return {k: (v if not isinstance(v, (bytes, bytearray)) else v.decode()) for k, v in request.items()}


def _position(ticket: int):
    with mt5_session.MT5_LOCK:
        found = mt5.positions_get(ticket=ticket)
    if not found:
        raise NotFoundError(f"No hay ninguna posicion abierta con ticket {ticket}.")
    return found[0]


def _market_price(symbol: str, order_type: int) -> float:
    with mt5_session.MT5_LOCK:
        tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise MT5Error(f"Sin cotizacion para {symbol}.", **mt5_session.last_error())
    return tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid


def _round(symbol: str, price: float | None) -> float | None:
    if price is None:
        return None
    info = mt5_session.ensure_symbol(symbol)
    return round(float(price), info.digits)


# ---------------------------------------------------------------------------
# Abrir
# ---------------------------------------------------------------------------
@router.post("/open", summary="Abrir posicion a mercado")
def open_position(body: models.OpenPositionRequest) -> dict:
    """Ejecuta BUY o SELL a mercado.

    El SL/TP se pueden dar de dos formas: como PRECIO (`sl`, `tp`) o como
    DISTANCIA en puntos (`sl_points`, `tp_points`). Si mandas ambas, gana el
    precio absoluto. Un punto es `symbol_info.point` (para EURUSD de 5 digitos,
    1 punto = 0,00001; 10 puntos = 1 pip)."""
    mt5_session.require_trading_allowed()
    symbol = body.symbol
    info = mt5_session.ensure_symbol(symbol)
    order_type = _SIDE_TO_ORDER[body.side]
    volume = mt5_session.normalize_volume(symbol, body.volume)
    price = _market_price(symbol, order_type)

    sl, tp = body.sl, body.tp
    point = info.point
    if sl is None and body.sl_points:
        sl = price - body.sl_points * point if body.side == "BUY" else price + body.sl_points * point
    if tp is None and body.tp_points:
        tp = price + body.tp_points * point if body.side == "BUY" else price - body.tp_points * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": _round(symbol, price),
        "deviation": body.deviation if body.deviation is not None else config.DEFAULT_DEVIATION,
        "magic": body.magic if body.magic is not None else config.DEFAULT_MAGIC,
        "comment": (body.comment or "")[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5_session.pick_filling_mode(symbol),
    }
    if sl:
        request["sl"] = _round(symbol, sl)
    if tp:
        request["tp"] = _round(symbol, tp)
    return _send(request)


# ---------------------------------------------------------------------------
# Cerrar
# ---------------------------------------------------------------------------
@router.post("/close/{ticket}", summary="Cerrar posicion (total o parcial)")
def close_position(
    body: models.ClosePositionRequest,
    ticket: int = Path(..., description="Ticket de la posicion abierta."),
) -> dict:
    """Sin `volume` cierra la posicion entera. Con `volume` menor que el de la
    posicion hace un CIERRE PARCIAL y el resto sigue abierto.

    En cuentas *hedging* el cierre parcial deja la posicion original con el
    volumen restante y su mismo ticket. En cuentas *netting* el resultado es
    una unica posicion neta."""
    mt5_session.require_trading_allowed()
    pos = _position(ticket)
    symbol = pos.symbol

    volume = pos.volume if body.volume is None else mt5_session.normalize_volume(symbol, body.volume)
    if volume > pos.volume + 1e-9:
        raise BadRequestError(
            f"El volumen a cerrar ({volume}) supera el de la posicion ({pos.volume}).",
            detail={"ticket": ticket, "position_volume": pos.volume})

    # Para cerrar se envia la orden CONTRARIA referenciando la posicion.
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": _round(symbol, _market_price(symbol, close_type)),
        "deviation": body.deviation if body.deviation is not None else config.DEFAULT_DEVIATION,
        "magic": pos.magic,
        "comment": (body.comment or "")[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5_session.pick_filling_mode(symbol),
    }
    out = _send(request)
    out["partial"] = volume < pos.volume - 1e-9
    out["remaining_volume"] = round(max(pos.volume - volume, 0.0), 8)
    return out


@router.post("/close-partial/{ticket}", summary="Cerrar parcialmente una posicion")
def close_partial(
    body: models.ClosePositionRequest,
    ticket: int = Path(..., description="Ticket de la posicion abierta."),
) -> dict:
    """Igual que `/trading/close/{ticket}` pero `volume` es OBLIGATORIO.

    Existe como endpoint propio porque es la operacion que mas se usa desde un
    cliente y asi el contrato es explicito."""
    if body.volume is None:
        raise BadRequestError("`volume` es obligatorio en un cierre parcial.")
    return close_position(body, ticket)


@router.post("/close-all", summary="Cerrar todas las posiciones (opcionalmente filtradas)")
def close_all(body: models.CloseAllRequest) -> dict:
    """Cierra en bloque. Devuelve el resultado de CADA cierre por separado: si
    una falla, las demas siguen adelante y el fallo queda registrado."""
    mt5_session.require_trading_allowed()
    with mt5_session.MT5_LOCK:
        raw = mt5.positions_get(symbol=body.symbol) if body.symbol else mt5.positions_get()
        raw = raw or ()
        targets = [p for p in raw if body.magic is None or p.magic == body.magic]

    results, closed, failed = [], 0, 0
    for pos in targets:
        try:
            res = close_position(models.ClosePositionRequest(comment=body.comment), pos.ticket)
            results.append({"ticket": pos.ticket, "symbol": pos.symbol, **res})
            closed += 1 if res.get("success") else 0
            failed += 0 if res.get("success") else 1
        except Exception as exc:  # noqa: BLE001 -- un fallo no debe abortar el resto
            failed += 1
            results.append({"ticket": pos.ticket, "symbol": pos.symbol, "success": False,
                            "error": getattr(exc, "detail", {"message": str(exc)})})
    return {"requested": len(targets), "closed": closed, "failed": failed, "results": results}


# ---------------------------------------------------------------------------
# Modificar SL / TP
# ---------------------------------------------------------------------------
def _modify_sltp(ticket: int, sl: float | None, tp: float | None) -> dict:
    mt5_session.require_trading_allowed()
    pos = _position(ticket)
    # `None` = no tocar (se conserva el valor actual). `0` = quitar.
    new_sl = pos.sl if sl is None else sl
    new_tp = pos.tp if tp is None else tp
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": ticket,
        "sl": _round(pos.symbol, new_sl) or 0.0,
        "tp": _round(pos.symbol, new_tp) or 0.0,
        "magic": pos.magic,
    }
    out = _send(request)
    out["previous"] = {"sl": pos.sl, "tp": pos.tp}
    return out


@router.post("/modify/{ticket}", summary="Mover SL y/o TP de una posicion")
def modify_sltp(body: models.ModifySLTPRequest, ticket: int = Path(...)) -> dict:
    """Omite un campo (`null`) para dejarlo como esta. Manda `0` para QUITARLO."""
    return _modify_sltp(ticket, body.sl, body.tp)


@router.post("/modify-sl/{ticket}", summary="Mover solo el Stop Loss")
def modify_sl(body: models.ModifySLRequest, ticket: int = Path(...)) -> dict:
    """El TP no se toca. `sl: 0` elimina el Stop Loss."""
    return _modify_sltp(ticket, body.sl, None)


@router.post("/modify-tp/{ticket}", summary="Mover solo el Take Profit")
def modify_tp(body: models.ModifyTPRequest, ticket: int = Path(...)) -> dict:
    """El SL no se toca. `tp: 0` elimina el Take Profit."""
    return _modify_sltp(ticket, None, body.tp)


# ---------------------------------------------------------------------------
# Ordenes pendientes
# ---------------------------------------------------------------------------
@router.post("/pending", summary="Colocar orden pendiente (limit / stop)")
def place_pending(body: models.PlacePendingRequest) -> dict:
    mt5_session.require_trading_allowed()
    symbol = body.symbol
    mt5_session.ensure_symbol(symbol)
    volume = mt5_session.normalize_volume(symbol, body.volume)

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": _PENDING_TO_ORDER[body.type],
        "price": _round(symbol, body.price),
        "magic": body.magic if body.magic is not None else config.DEFAULT_MAGIC,
        "comment": (body.comment or "")[:31],
        "type_filling": mt5.ORDER_FILLING_RETURN,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if body.sl:
        request["sl"] = _round(symbol, body.sl)
    if body.tp:
        request["tp"] = _round(symbol, body.tp)
    if body.stoplimit:
        request["stoplimit"] = _round(symbol, body.stoplimit)
    if body.expiration:
        request["type_time"] = mt5.ORDER_TIME_SPECIFIED
        request["expiration"] = body.expiration
    return _send(request)


@router.post("/pending/{ticket}/modify", summary="Modificar una orden pendiente")
def modify_pending(body: models.ModifyPendingRequest, ticket: int = Path(...)) -> dict:
    """Cambia precio de activacion, SL, TP o caducidad. Los campos `null` se
    dejan como estan."""
    mt5_session.require_trading_allowed()
    with mt5_session.MT5_LOCK:
        found = mt5.orders_get(ticket=ticket)
    if not found:
        raise NotFoundError(f"No hay ninguna orden pendiente con ticket {ticket}.")
    order = found[0]

    request = {
        "action": mt5.TRADE_ACTION_MODIFY,
        "order": ticket,
        "symbol": order.symbol,
        "price": _round(order.symbol, body.price if body.price is not None else order.price_open),
        "sl": _round(order.symbol, body.sl if body.sl is not None else order.sl) or 0.0,
        "tp": _round(order.symbol, body.tp if body.tp is not None else order.tp) or 0.0,
        "type_time": order.type_time,
        "expiration": body.expiration if body.expiration is not None else order.time_expiration,
    }
    if body.stoplimit is not None:
        request["stoplimit"] = _round(order.symbol, body.stoplimit)
    out = _send(request)
    out["previous"] = {"price": order.price_open, "sl": order.sl, "tp": order.tp}
    return out


@router.delete("/pending/{ticket}", summary="Cancelar una orden pendiente")
def cancel_pending(ticket: int = Path(...)) -> dict:
    mt5_session.require_trading_allowed()
    with mt5_session.MT5_LOCK:
        found = mt5.orders_get(ticket=ticket)
    if not found:
        raise NotFoundError(f"No hay ninguna orden pendiente con ticket {ticket}.")
    return _send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})


# ---------------------------------------------------------------------------
# Calculos previos (no envian nada al broker)
# ---------------------------------------------------------------------------
@router.post("/calc-margin", summary="Margen requerido por una operacion")
def calc_margin(body: models.CalcMarginRequest) -> dict:
    """Cuanto margen bloquearia la operacion. No envia nada al mercado."""
    mt5_session.require_connection()
    mt5_session.ensure_symbol(body.symbol)
    order_type = _SIDE_TO_ORDER[body.side]
    price = body.price if body.price is not None else _market_price(body.symbol, order_type)
    with mt5_session.MT5_LOCK:
        margin = mt5.order_calc_margin(order_type, body.symbol, body.volume, price)
        if margin is None:
            raise MT5Error("order_calc_margin fallo.", **mt5_session.last_error())
        currency = mt5.account_info().currency
    return {"symbol": body.symbol, "side": body.side, "volume": body.volume,
            "price": price, "margin": margin, "currency": currency}


@router.post("/calc-profit", summary="Beneficio teorico de una operacion")
def calc_profit(body: models.CalcProfitRequest) -> dict:
    """P&L que daria ir de `price_open` a `price_close`. No envia nada."""
    mt5_session.require_connection()
    mt5_session.ensure_symbol(body.symbol)
    with mt5_session.MT5_LOCK:
        profit = mt5.order_calc_profit(_SIDE_TO_ORDER[body.side], body.symbol, body.volume,
                                       body.price_open, body.price_close)
        if profit is None:
            raise MT5Error("order_calc_profit fallo.", **mt5_session.last_error())
        currency = mt5.account_info().currency
    return {"symbol": body.symbol, "side": body.side, "volume": body.volume,
            "price_open": body.price_open, "price_close": body.price_close,
            "profit": profit, "currency": currency}


@router.post("/check", summary="Validar una orden SIN enviarla")
def check_order(body: models.OpenPositionRequest) -> dict:
    """Simula la orden contra el servidor: dice si pasaria, cuanto margen
    consumiria y como quedaria el balance. **No envia nada al mercado.**

    Es la forma segura de validar desde .NET antes de ejecutar de verdad."""
    mt5_session.require_connection()
    symbol = body.symbol
    mt5_session.ensure_symbol(symbol)
    order_type = _SIDE_TO_ORDER[body.side]
    volume = mt5_session.normalize_volume(symbol, body.volume)
    price = _market_price(symbol, order_type)
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
        "type": order_type, "price": _round(symbol, price),
        "deviation": body.deviation if body.deviation is not None else config.DEFAULT_DEVIATION,
        "magic": body.magic if body.magic is not None else config.DEFAULT_MAGIC,
        "comment": (body.comment or "")[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5_session.pick_filling_mode(symbol),
    }
    if body.sl:
        request["sl"] = _round(symbol, body.sl)
    if body.tp:
        request["tp"] = _round(symbol, body.tp)
    with mt5_session.MT5_LOCK:
        result = mt5.order_check(request)
        if result is None:
            raise MT5Error("order_check() devolvio None.", **mt5_session.last_error())
        out = converters.order_result_to_dict(result, is_check=True)
    out["request"] = _jsonable(request)
    return out
