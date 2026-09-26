"""Modelos Pydantic de entrada y salida.

Los cuerpos de peticion estan tipados para que FastAPI genere el esquema
OpenAPI (`/openapi.json`), del que el cliente .NET puede generar sus clases
automaticamente (NSwag / Refitter / `dotnet openapi`).

Las respuestas de lectura se devuelven como `dict` sin modelo estricto: los
campos de MT5 varian entre brokers y builds del terminal, y forzar un esquema
cerrado haria que el gateway PIERDA campos que el broker si esta enviando.
El contrato real de cada respuesta esta documentado en el PDF."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["BUY", "SELL"]
PendingType = Literal["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"]


class OpenPositionRequest(BaseModel):
    symbol: str = Field(..., examples=["EURUSD"], description="Nombre exacto del simbolo en el broker.")
    side: Side = Field(..., description="BUY o SELL. Ejecucion a mercado.")
    volume: float = Field(..., gt=0, examples=[0.10], description="Lotes. Se ajusta al volume_step del simbolo, siempre hacia abajo.")
    sl: float | None = Field(None, description="Stop Loss como PRECIO absoluto. null = sin SL.")
    tp: float | None = Field(None, description="Take Profit como PRECIO absoluto. null = sin TP.")
    sl_points: int | None = Field(None, description="Alternativa a `sl`: distancia en puntos desde el precio de entrada. Se ignora si `sl` viene informado.")
    tp_points: int | None = Field(None, description="Alternativa a `tp`: distancia en puntos desde el precio de entrada.")
    deviation: int | None = Field(None, description="Slippage maximo aceptado, en puntos. Por defecto el del gateway (20).")
    magic: int | None = Field(None, description="Magic number de la orden. Por defecto el del gateway.")
    comment: str | None = Field(None, max_length=31, description="Comentario libre. MT5 lo trunca a 31 caracteres.")

    model_config = {"json_schema_extra": {"examples": [
        {"symbol": "EURUSD", "side": "BUY", "volume": 0.10, "sl": 1.0850, "tp": 1.0950,
         "deviation": 20, "comment": "desde .NET"}
    ]}}


class ClosePositionRequest(BaseModel):
    volume: float | None = Field(None, gt=0, description="Volumen a cerrar. null o ausente = cerrar la posicion ENTERA.")
    deviation: int | None = Field(None, description="Slippage maximo en puntos.")
    comment: str | None = Field(None, max_length=31)

    model_config = {"json_schema_extra": {"examples": [{"volume": 0.05, "comment": "cierre parcial"}]}}


class ModifySLTPRequest(BaseModel):
    sl: float | None = Field(None, description="Nuevo Stop Loss (precio absoluto). Enviar 0 para QUITAR el SL. Omitir (null) para dejarlo como esta.")
    tp: float | None = Field(None, description="Nuevo Take Profit (precio absoluto). Enviar 0 para QUITAR el TP. Omitir (null) para dejarlo como esta.")

    model_config = {"json_schema_extra": {"examples": [{"sl": 1.0860, "tp": 1.0990}]}}


class ModifySLRequest(BaseModel):
    sl: float = Field(..., description="Nuevo Stop Loss como precio absoluto. 0 = quitar el SL.")
    model_config = {"json_schema_extra": {"examples": [{"sl": 1.0860}]}}


class ModifyTPRequest(BaseModel):
    tp: float = Field(..., description="Nuevo Take Profit como precio absoluto. 0 = quitar el TP.")
    model_config = {"json_schema_extra": {"examples": [{"tp": 1.0990}]}}


class PlacePendingRequest(BaseModel):
    symbol: str = Field(..., examples=["EURUSD"])
    type: PendingType = Field(..., description="Tipo de orden pendiente.")
    volume: float = Field(..., gt=0, examples=[0.10])
    price: float = Field(..., description="Precio de activacion de la orden.")
    sl: float | None = None
    tp: float | None = None
    stoplimit: float | None = Field(None, description="Solo para BUY_STOP_LIMIT / SELL_STOP_LIMIT.")
    expiration: int | None = Field(None, description="Epoch en segundos (hora de servidor). null = GTC (sin caducidad).")
    magic: int | None = None
    comment: str | None = Field(None, max_length=31)

    model_config = {"json_schema_extra": {"examples": [
        {"symbol": "EURUSD", "type": "BUY_LIMIT", "volume": 0.10, "price": 1.0800,
         "sl": 1.0750, "tp": 1.0900}
    ]}}


class ModifyPendingRequest(BaseModel):
    price: float | None = Field(None, description="Nuevo precio de activacion. null = dejarlo igual.")
    sl: float | None = None
    tp: float | None = None
    stoplimit: float | None = None
    expiration: int | None = None

    model_config = {"json_schema_extra": {"examples": [{"price": 1.0790, "sl": 1.0740}]}}


class CloseAllRequest(BaseModel):
    symbol: str | None = Field(None, description="Cerrar solo las posiciones de este simbolo. null = todas.")
    magic: int | None = Field(None, description="Cerrar solo las posiciones con este magic. null = cualquiera.")
    comment: str | None = Field(None, max_length=31)

    model_config = {"json_schema_extra": {"examples": [{"symbol": "EURUSD"}]}}


class CalcMarginRequest(BaseModel):
    symbol: str = Field(..., examples=["EURUSD"])
    side: Side
    volume: float = Field(..., gt=0)
    price: float | None = Field(None, description="Precio de referencia. null = precio de mercado actual.")


class CalcProfitRequest(BaseModel):
    symbol: str = Field(..., examples=["EURUSD"])
    side: Side
    volume: float = Field(..., gt=0)
    price_open: float
    price_close: float


class InternalLoginRequest(BaseModel):
    """Credenciales que el gateway pasa a un worker del pool.

    Nunca se persisten: el worker las usa para autenticar su terminal y las
    descarta. No aparecen en ningun log."""
    login: int = Field(..., description="Numero de cuenta MT5.")
    password: str = Field(..., description="Contrasena de la cuenta (de inversor o de operador).")
    server: str = Field(..., description="Nombre exacto del servidor del broker.")
