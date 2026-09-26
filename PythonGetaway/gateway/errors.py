"""Errores del gateway, con el MISMO formato que ya devuelven los workers.

Da igual que el fallo lo genere el gateway (token invalido, pool agotado) o el
worker (error de MT5): el cliente .NET deserializa siempre la misma forma.

    {"error": {"code": "...", "message": "...", "mt5_code": null, "detail": {...}}}
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
