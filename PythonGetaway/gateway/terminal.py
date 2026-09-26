"""Control explicito del terminal MT5 del usuario.

Estos endpoints hablan en terminos de TERMINAL en vez de en terminos de sesion.
Por debajo son la misma operacion que `/auth/login` y `/auth/logout` -- en esta
arquitectura una sesion ES un terminal dedicado -- pero se exponen con nombres
explicitos porque es como razona el usuario: "abre mi MT5", "cierra mi MT5".

    POST /terminal/open    == POST /auth/login
    POST /terminal/close   == POST /auth/logout

La logica vive UNA sola vez, en `gateway/auth.py`, para que no puedan divergir.

CICLO DE VIDA
-------------
El terminal se mantiene vivo MIENTRAS HAYA ACTIVIDAD: cualquier peticion
autenticada reinicia el contador. Si pasan `PYGW_SESSION_IDLE` segundos (2 h por
defecto) sin una sola peticion, el barrido lo cierra solo y libera el slot; a
partir de ahi el usuario tiene que volver a abrirlo con sus credenciales.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from gateway import config
from gateway.auth import (LoginRequest, LoginResponse, LogoutResponse, bearer_token,
                          close_terminal, current_session, open_terminal)

router = APIRouter(prefix="/terminal", tags=["terminal"])


class TerminalStatusResponse(BaseModel):
    """Estado del terminal MT5 de la sesion, comprobado EN VIVO contra el worker."""
    open: bool = Field(..., description="`true` si el terminal esta abierto y asignado a esta sesion.")
    connected: bool = Field(..., description="`true` si ademas tiene conexion con el servidor del "
                                             "broker. Un terminal abierto puede estar sin conexion.")
    slot_id: str = Field(..., description="Terminal asignado.", examples=["slot-1"])
    slot_state: str = Field(..., description="Estado del slot en el pool.", examples=["BUSY"])
    login: int = Field(..., description="Cuenta autenticada en el terminal.", examples=[5054767214])
    server: str = Field(..., examples=["MetaQuotes-Demo"])
    trade_allowed: bool | None = Field(None, description="Si el terminal permite operar ahora mismo.")
    opened_at: int = Field(..., description="Epoch UTC en que se abrio el terminal.",
                           examples=[1789040000])
    last_activity: int = Field(..., description="Epoch UTC de la ultima peticion autenticada.",
                               examples=[1789040420])
    idle_seconds: int = Field(..., description="Segundos transcurridos desde la ultima peticion.",
                              examples=[420])
    closes_in: int = Field(..., description="Segundos que faltan para el cierre automatico si no "
                                            "llega ninguna peticion mas.", examples=[6780])
    closes_at: int = Field(..., description="Epoch UTC del cierre automatico previsto.",
                           examples=[1789047200])
    idle_timeout_seconds: int = Field(..., description="Inactividad tolerada antes de cerrar.",
                                      examples=[7200])
    ttl_seconds: int = Field(..., description="Tope absoluto de vida. `0` = sin tope: solo cuenta "
                                              "la inactividad.", examples=[0])


@router.post("/open", summary="Abrir el terminal MT5 con credenciales propias",
             response_model=LoginResponse)
async def terminal_open(body: LoginRequest, request: Request) -> dict:
    """Arranca una sesion MT5 dedicada para esta cuenta y devuelve el token.

    Es **la misma operacion** que `POST /auth/login`; existe con este nombre
    para quien prefiere razonar en terminos de terminal.

    Si la cuenta ya tiene un terminal abierto, se devuelve **ese mismo**
    (`reused: true`) en lugar de consumir un segundo slot del pool.

    A partir de aqui el terminal sigue vivo mientras el cliente haga peticiones.
    """
    return await open_terminal(body, request)


@router.post("/close", summary="Cerrar el terminal MT5 de forma explicita",
             response_model=LogoutResponse)
async def terminal_close(request: Request, token: str = Depends(bearer_token)) -> dict:
    """Cierra el terminal y devuelve el slot al pool.

    Es **la misma operacion** que `POST /auth/logout`.

    El token queda invalidado de inmediato. Por defecto el terminal ademas se
    reinicia para que no quede ninguna cuenta conectada
    (`PYGW_RECYCLE_ON_LOGOUT`), asi que el slot tarda unos segundos en volver a
    estar disponible para otro usuario.
    """
    return await close_terminal(request, token)


@router.get("/status", summary="Estado del terminal y cuanto le queda vivo",
            response_model=TerminalStatusResponse)
async def terminal_status(request: Request, session=Depends(current_session)) -> dict:
    """Dice si el terminal esta abierto, si tiene conexion con el broker y
    **cuantos segundos faltan para el cierre automatico**.

    Ojo: llamar a este endpoint cuenta como actividad, luego tambien reinicia el
    contador de inactividad. Sirve como latido si el cliente esta parado.
    """
    pool = request.app.state.pool
    slot = next((s for s in pool.slots if s.slot_id == session.slot_id), None)

    connected, trade_allowed = False, None
    if slot is not None and slot.state == "BUSY":
        # Se pregunta al worker por el estado REAL del terminal: que el slot
        # este marcado BUSY no garantiza que siga conectado con el broker.
        try:
            r = await pool.client.get(f"{slot.base_url}/health", timeout=10)
            if r.status_code == 200:
                data = r.json()
                connected = bool(data.get("connected"))
                trade_allowed = (data.get("account") or {}).get("trade_allowed")
        except Exception:  # noqa: BLE001 -- un worker mudo no debe romper el estado
            connected = False

    now = time.time()
    return {
        "open": slot is not None and slot.state == "BUSY",
        "connected": connected,
        "slot_id": session.slot_id,
        "slot_state": "GONE" if slot is None else slot.state,
        "login": session.login,
        "server": session.server,
        "trade_allowed": trade_allowed,
        "opened_at": int(session.created_at),
        "last_activity": int(session.last_seen),
        "idle_seconds": max(0, int(now - session.last_seen)),
        "closes_in": max(0, int(session.expires_at() - now)),
        "closes_at": int(session.expires_at()),
        "idle_timeout_seconds": config.SESSION_IDLE_TIMEOUT,
        "ttl_seconds": config.SESSION_TTL,
    }
