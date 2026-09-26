"""Autenticacion: /auth/login, /auth/logout, /auth/session.

Cada usuario aporta SUS credenciales MT5. El gateway le reserva un slot del
pool (terminal propio), lo autentica y devuelve un token. A partir de ahi, todo
el resto de la API se llama con `Authorization: Bearer <token>`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from gateway import config
from gateway.errors import GatewayError
from gateway.pool import LoginFailed, PoolExhausted, WorkerUnreachable

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("pygw.auth")


class LoginRequest(BaseModel):
    login: int = Field(..., description="Numero de cuenta MT5.", examples=[5054767214])
    password: str = Field(..., description="Contrasena de la cuenta (de operador o de inversor). "
                                           "No se guarda en ningun sitio: se usa para autenticar "
                                           "el terminal y se descarta.")
    server: str = Field(..., description="Nombre exacto del servidor del broker, tal cual aparece "
                                         "en MT5. Distingue mayusculas.",
                        examples=["MetaQuotes-Demo"])

    model_config = {"json_schema_extra": {"examples": [
        {"login": 5054767214, "password": "<tu contrasena>", "server": "MetaQuotes-Demo"}
    ]}}


# --- Respuestas tipadas ----------------------------------------------------
# Se declaran como modelos (y no como `dict`) para que los tipos entren en el
# esquema OpenAPI: asi NSwag/Refitter generan las clases C# correctas y el PDF
# puede documentar automaticamente que devuelve cada endpoint.

class AccountSummary(BaseModel):
    """Resumen de la cuenta autenticada. Los datos COMPLETOS estan en `GET /account`."""
    login: int = Field(..., description="Numero de cuenta.", examples=[5054767214])
    server: str = Field(..., description="Servidor del broker.", examples=["MetaQuotes-Demo"])
    currency: str = Field(..., description="Divisa de la cuenta.", examples=["USD"])
    balance: float = Field(..., description="Saldo.", examples=[96126.17])
    equity: float = Field(..., description="Fondos (saldo + flotante).", examples=[96126.17])
    trade_mode: int = Field(..., description="0 = DEMO, 1 = CONTEST, 2 = REAL.", examples=[0])
    trade_mode_name: str = Field(..., description="El mismo valor en texto.", examples=["DEMO"])
    trade_allowed: bool = Field(..., description="Si el broker permite operar en esta cuenta.")


class SessionInfo(BaseModel):
    """Datos de una sesion. `GET /auth/session` devuelve esto (sin `token`)."""
    slot_id: str = Field(..., description="Terminal MT5 asignado a esta sesion.", examples=["slot-1"])
    login: int = Field(..., description="Cuenta autenticada.", examples=[5054767214])
    server: str = Field(..., examples=["MetaQuotes-Demo"])
    account: AccountSummary | None = Field(None, description="Resumen de la cuenta.")
    created_at: int = Field(..., description="Epoch UTC en segundos del login.", examples=[1789040000])
    last_seen: int = Field(..., description="Epoch UTC de la ultima peticion con este token.",
                           examples=[1789040420])
    expires_at: int = Field(..., description="Epoch UTC en que caduca. Es el MINIMO entre vida "
                                             "maxima e inactividad.", examples=[1789041320])
    expires_in: int = Field(..., description="Segundos que le quedan.", examples=[900])
    ttl_seconds: int = Field(..., description="Vida maxima configurada.", examples=[3600])
    idle_timeout_seconds: int = Field(..., description="Ventana de inactividad configurada.",
                                      examples=[900])


class LoginResponse(SessionInfo):
    """Respuesta de `POST /auth/login`."""
    token: str = Field(..., description="Token de sesion. Va en `Authorization: Bearer <token>` "
                                        "en TODAS las llamadas siguientes.",
                       examples=["kHs9_2Fq...Zx1"])
    reused: bool = Field(..., description="`true` si la cuenta ya tenia una sesion viva y se "
                                          "devuelve esa misma en lugar de consumir otro terminal.")


class LogoutResponse(BaseModel):
    logged_out: bool = Field(..., description="Siempre `true` si el token era valido.")
    recycled: bool = Field(..., description="Si el terminal se reinicia para no dejar ninguna "
                                            "cuenta conectada (`PYGW_RECYCLE_ON_LOGOUT`).")


class SessionListItem(BaseModel):
    slot_id: str
    login: int
    server: str
    created_at: int
    last_seen: int
    expires_at: int
    expires_in: int
    ttl_seconds: int
    idle_timeout_seconds: int


class SessionListResponse(BaseModel):
    count: int = Field(..., description="Numero de sesiones activas.")
    sessions: list[SessionListItem] = Field(..., description="Una entrada por sesion. "
                                                             "No expone tokens ni credenciales.")


def bearer_token(authorization: str | None = Header(None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise GatewayError(401, "MISSING_TOKEN",
                           "Falta la cabecera 'Authorization: Bearer <token>'. "
                           "Obten un token en POST /auth/login.")
    return authorization.split(" ", 1)[1].strip()


def current_session(request: Request, token: str = Depends(bearer_token)):
    session = request.app.state.sessions.get(token)
    if session is None:
        raise GatewayError(401, "INVALID_OR_EXPIRED_TOKEN",
                           "El token no existe o ha caducado. Vuelve a autenticarte.")
    return session


async def open_terminal(body: LoginRequest, request: Request) -> dict:
    """Logica compartida por `POST /auth/login` y `POST /terminal/open`.

    Son la MISMA operacion (reservar un terminal y autenticarlo); existen con
    los dos nombres porque unos clientes razonan en terminos de sesion y otros
    en terminos de terminal. Se implementa una sola vez para que no puedan
    divergir."""
    sessions = request.app.state.sessions
    pool = request.app.state.pool
    key = f"{body.login}@{body.server}"

    if sessions.too_many_attempts(key):
        raise GatewayError(429, "TOO_MANY_LOGIN_ATTEMPTS",
                           f"Demasiados intentos fallidos para esta cuenta. "
                           f"Espera {config.LOGIN_ATTEMPT_WINDOW} segundos.")

    existing = sessions.find_by_account(body.login, body.server)
    if existing is not None:
        log.info("reutilizando sesion existente para la cuenta %s", body.login)
        return {"reused": True, **existing.public(include_token=True)}

    sessions.note_attempt(key)
    try:
        slot = await pool.acquire(body.login, body.password, body.server)
    except PoolExhausted as exc:
        raise GatewayError(503, "POOL_EXHAUSTED", str(exc),
                           detail=pool.status()) from exc
    except WorkerUnreachable as exc:
        raise GatewayError(502, "WORKER_UNREACHABLE", str(exc)) from exc
    except LoginFailed as exc:
        # El error del worker ya viene con el codigo de MT5; se reenvia tal cual
        # para que el cliente sepa si fue contrasena, servidor o cuenta.
        raise GatewayError(401, "MT5_LOGIN_FAILED",
                           "MT5 rechazo las credenciales o no pudo conectar con el servidor.",
                           detail=exc.payload) from exc

    account = slot.account or {}
    if account.get("trade_mode_name") == "REAL" and not config.ALLOW_REAL_ACCOUNT:
        log.warning("cuenta REAL %s autenticada: el trading quedara bloqueado (403)", body.login)

    sessions.clear_attempts(key)
    session = await sessions.create(slot, body.login, body.server)
    return {"reused": False, **session.public(include_token=True)}


async def close_terminal(request: Request, token: str) -> dict:
    """Logica compartida por `POST /auth/logout` y `POST /terminal/close`."""
    dropped = await request.app.state.sessions.drop(token)
    if not dropped:
        raise GatewayError(401, "INVALID_OR_EXPIRED_TOKEN",
                           "El token no existe o ya habia caducado.")
    return {"logged_out": True, "recycled": config.RECYCLE_ON_LOGOUT}


@router.post("/logout", summary="Cerrar sesion y liberar el terminal",
             response_model=LogoutResponse)
async def logout(request: Request, token: str = Depends(bearer_token)) -> dict:
    """Cierra la sesion MT5 y devuelve el slot al pool.

    Identico a `POST /terminal/close`: misma operacion, dos nombres.

    Por defecto el terminal se reinicia para que no quede ninguna cuenta
    conectada (`PYGW_RECYCLE_ON_LOGOUT`), asi que el slot tarda unos segundos en
    volver a estar disponible."""
    return await close_terminal(request, token)


@router.post("/login", summary="Autenticarse en MT5 con credenciales propias",
             response_model=LoginResponse)
async def login(body: LoginRequest, request: Request) -> dict:
    """Reserva un terminal MT5 dedicado para esta cuenta y devuelve un token.

    Identico a `POST /terminal/open`: misma operacion, dos nombres.

    Si la cuenta ya tiene una sesion viva, se devuelve **la misma sesion** en
    lugar de consumir un segundo slot.

    La contrasena no se guarda: viaja una vez hasta el worker, autentica el
    terminal y se descarta. Nunca aparece en logs.
    """
    return await open_terminal(body, request)


@router.post("/refresh", summary="Renovar la sesion sin reenviar credenciales",
             response_model=SessionInfo)
async def refresh(session=Depends(current_session)) -> dict:
    """Reinicia el reloj de **vida maxima** de la sesion.

    El token **no cambia**: se sigue usando el mismo, con lo que no hay carrera
    con peticiones en vuelo. Solo se reinicia el contador.

    Cuando hace falta:

    - La ventana de **inactividad** (15 min) ya se renueva sola con cualquier
      llamada autenticada. Para eso no hace falta este endpoint.
    - El tope de **vida maxima** (1 h) NO se renueva solo. Un cliente que este
      conectado toda la sesion bursatil debe llamar aqui cada cierto tiempo
      (p.ej. cada 30 min) para no verse obligado a reenviar las credenciales.

    Si prefieres que no exista tope absoluto, arranca con `PYGW_SESSION_TTL=0`:
    entonces la sesion solo muere por inactividad y este endpoint es opcional.
    """
    session.renew()
    return session.public()


@router.get("/session", summary="Estado de la sesion actual",
            response_model=SessionInfo)
async def session_info(session=Depends(current_session)) -> dict:
    """Datos de la sesion asociada al token, con lo que le queda de vida.
    Llamarlo tambien **renueva** la ventana de inactividad."""
    return session.public()


@router.get("/sessions", summary="Sesiones activas (sin datos sensibles)",
            response_model=SessionListResponse)
async def list_sessions(request: Request) -> dict:
    """Vision de operacion: cuantas sesiones hay y en que slot esta cada una.
    No expone tokens ni credenciales."""
    items = []
    for s in request.app.state.sessions.all():
        data = s.public()
        data.pop("account", None)
        items.append(data)
    return {"count": len(items), "sessions": items}
