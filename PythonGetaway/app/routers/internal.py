"""Endpoints INTERNOS worker <-> gateway.

No forman parte de la API publica: el gateway nunca los proxifica hacia fuera
(`gateway/proxy.py` los tiene en la lista negra). Solo escuchan en localhost y
exigen el secreto compartido `PYGW_INTERNAL_TOKEN`.

Las credenciales llegan aqui una unica vez, se usan para autenticar el terminal
y se descartan. No se guardan en memoria persistente, ni en disco, ni se
escriben en el log."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header

from app import config, mt5_session
from app.errors import GatewayError
from app.models import InternalLoginRequest

router = APIRouter(prefix="/internal", tags=["internal"], include_in_schema=False)
log = logging.getLogger("pygw.worker")


def _check_secret(token: str | None) -> None:
    """El worker escucha en 127.0.0.1, pero cualquier proceso local podria
    hablarle. El secreto impide que otro proceso secuestre un slot ya
    autenticado y opere con la cuenta de un usuario."""
    if config.INTERNAL_TOKEN and token != config.INTERNAL_TOKEN:
        raise GatewayError(401, "BAD_INTERNAL_TOKEN",
                           "Secreto interno invalido para hablar con este worker.")


@router.get("/status")
def internal_status(x_internal_token: str | None = Header(None)) -> dict:
    _check_secret(x_internal_token)
    state = mt5_session.status()
    account = state.get("account") or {}
    return {
        "slot_id": config.SLOT_ID,
        "worker_mode": config.WORKER_MODE,
        "terminal_path": config.TERMINAL_PATH,
        "busy": bool(state.get("initialized")),
        "login": account.get("login"),
        "server": account.get("server"),
        "trade_mode_name": account.get("trade_mode_name"),
        "connected": state.get("connected"),
    }


@router.post("/login")
def internal_login(body: InternalLoginRequest,
                   x_internal_token: str | None = Header(None)) -> dict:
    _check_secret(x_internal_token)
    # Se registra la cuenta y el servidor, NUNCA la contrasena.
    log.info("slot=%s autenticando cuenta %s en %s", config.SLOT_ID, body.login, body.server)
    state = mt5_session.login_account(body.login, body.password, body.server)
    account = state.get("account") or {}
    log.info("slot=%s autenticado: login=%s tipo=%s", config.SLOT_ID,
             account.get("login"), account.get("trade_mode_name"))
    return {"slot_id": config.SLOT_ID, "account": account, "terminal": state.get("terminal")}


@router.post("/logout")
def internal_logout(x_internal_token: str | None = Header(None)) -> dict:
    _check_secret(x_internal_token)
    log.info("slot=%s cerrando sesion", config.SLOT_ID)
    return mt5_session.logout_account()
