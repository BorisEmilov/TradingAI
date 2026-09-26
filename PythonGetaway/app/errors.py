"""Errores del gateway con un formato JSON unico.

Todo fallo -- de MT5, de validacion o de logica -- sale con la misma forma, para
que el cliente .NET pueda deserializarlo siempre al mismo tipo:

    {"error": {"code": "MT5_ERROR", "message": "...", "mt5_code": -4, "detail": {...}}}
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


class GatewayError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str,
                 mt5_code: int | None = None, detail: Any = None):
        super().__init__(status_code=status_code, detail={
            "error": {"code": code, "message": message, "mt5_code": mt5_code, "detail": detail}
        })


class MT5Error(GatewayError):
    """Fallo devuelto por la propia libreria MetaTrader5."""

    def __init__(self, message: str, mt5_code: int | None = None, detail: Any = None,
                 status_code: int = 502):
        super().__init__(status_code, "MT5_ERROR", message, mt5_code, detail)


class NotConnectedError(GatewayError):
    def __init__(self, message: str = "El gateway no esta conectado al terminal MT5."):
        super().__init__(503, "NOT_CONNECTED", message)


class NotFoundError(GatewayError):
    def __init__(self, message: str, detail: Any = None):
        super().__init__(404, "NOT_FOUND", message, detail=detail)


class BadRequestError(GatewayError):
    def __init__(self, message: str, detail: Any = None):
        super().__init__(400, "BAD_REQUEST", message, detail=detail)


class RealAccountBlockedError(GatewayError):
    """Guardarrail: la cuenta conectada no es DEMO y no se autorizo operar en real."""

    def __init__(self, trade_mode: int, login: int | None = None):
        super().__init__(
            403, "REAL_ACCOUNT_BLOCKED",
            "La cuenta conectada no es DEMO y el gateway arranco sin permiso para operar "
            "en real. Los endpoints de lectura siguen disponibles. Para permitir el envio "
            "de ordenes, arranca con PYGW_ALLOW_REAL=1.",
            detail={"trade_mode": trade_mode, "login": login},
        )
