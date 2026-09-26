"""Cuenta, terminal y estado de la conexion."""

from __future__ import annotations

import MetaTrader5 as mt5
from fastapi import APIRouter

from app import config, converters, mt5_session
from app.errors import MT5Error

router = APIRouter(tags=["account"])


@router.get("/health", summary="Estado del gateway y del terminal")
def health() -> dict:
    """No requiere conexion: sirve para saber SI hay conexion.

    Es el unico endpoint que nunca devuelve 5xx por estar desconectado."""
    return mt5_session.status()


@router.get("/account", summary="Datos completos de la cuenta")
def account() -> dict:
    """Balance, equity, margen, apalancamiento, divisa, tipo de cuenta...

    `trade_mode` 0=DEMO 1=CONTEST 2=REAL (tambien como `trade_mode_name`)."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        info = mt5.account_info()
        if info is None:
            raise MT5Error("account_info() devolvio None.", **mt5_session.last_error())
        return converters.account_to_dict(info)


@router.get("/terminal", summary="Datos del terminal MT5")
def terminal() -> dict:
    """Build, rutas, permisos (trade_allowed, dlls_allowed), ping, trafico..."""
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        info = mt5.terminal_info()
        if info is None:
            raise MT5Error("terminal_info() devolvio None.", **mt5_session.last_error())
        data = converters.named_to_dict(info)
        version = mt5.version()
        if version:
            data["api_version"] = version[0]
            data["api_build"] = version[1]
            data["api_build_date"] = version[2]
        return data


@router.get("/version", summary="Version de la API de MetaTrader5")
def version() -> dict:
    mt5_session.require_connection()
    with mt5_session.MT5_LOCK:
        v = mt5.version()
        if v is None:
            raise MT5Error("version() devolvio None.", **mt5_session.last_error())
        return {"version": v[0], "build": v[1], "build_date": v[2]}


@router.get("/config", summary="Configuracion efectiva del gateway")
def gateway_config() -> dict:
    """Lo que el gateway tiene cargado. Nunca expone la contrasena."""
    return {
        "host": config.HOST,
        "port": config.PORT,
        "terminal_path": config.TERMINAL_PATH,
        "login_configured": config.MT5_LOGIN is not None,
        "server_configured": config.MT5_SERVER,
        "allow_real_account": config.ALLOW_REAL_ACCOUNT,
        "default_magic": config.DEFAULT_MAGIC,
        "default_deviation": config.DEFAULT_DEVIATION,
        "max_bars": config.MAX_BARS,
        "max_ticks": config.MAX_TICKS,
    }


@router.get("/last-error", summary="Ultimo error de la libreria MT5")
def last_error() -> dict:
    return mt5_session.last_error()
