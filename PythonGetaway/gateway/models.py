"""Modelos de respuesta de los endpoints propios del gateway.

Se declaran tipados (y no como `dict`) para que los tipos entren en el esquema
OpenAPI: asi NSwag/Refitter generan las clases C# correctas y la documentacion
puede listar automaticamente que devuelve cada endpoint y con que tipo."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SlotInfo(BaseModel):
    """Un terminal MT5 del pool."""
    slot_id: str = Field(..., description="Identificador del slot.", examples=["slot-1"])
    state: str = Field(..., description="COLD | WARMING | IDLE | BUSY | RECYCLING | FAILED. "
                                        "Solo los IDLE aceptan un login nuevo.",
                       examples=["IDLE"])
    port: int = Field(..., description="Puerto local del worker. Uso interno; no se expone fuera.",
                      examples=[8101])
    login: int | None = Field(None, description="Cuenta autenticada en este slot, o null si esta libre.")
    server: str | None = Field(None, description="Servidor del broker, o null si esta libre.")
    seconds_in_state: float = Field(..., description="Segundos que lleva en el estado actual.",
                                    examples=[12.4])
    error: str | None = Field(None, description="Motivo del fallo si el estado es FAILED.")


class PoolStatus(BaseModel):
    size: int = Field(..., description="Numero total de slots (usuarios concurrentes maximos).",
                      examples=[3])
    states: dict[str, int] = Field(..., description="Recuento de slots por estado.",
                                   examples=[{"IDLE": 2, "BUSY": 1}])
    available: int = Field(..., description="Slots en IDLE: cuantos usuarios NUEVOS pueden "
                                            "autenticarse ahora mismo.", examples=[2])
    slots: list[SlotInfo] = Field(..., description="Detalle de cada slot.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="`ok` si queda algun slot libre, `saturado` si no.",
                        examples=["ok"])
    pool: PoolStatus
    sessions_active: int = Field(..., description="Sesiones de usuario vivas.", examples=[1])
    allow_real_account: bool = Field(..., description="Si el trading esta permitido en cuentas REALES.")
    recycle_on_logout: bool = Field(..., description="Si el terminal se reinicia al cerrar sesion.")
    session_ttl_seconds: int = Field(..., description="Vida maxima de un token.", examples=[3600])
    session_idle_timeout_seconds: int = Field(..., description="Caducidad por inactividad.",
                                              examples=[900])


class GatewayIndexResponse(BaseModel):
    name: str = Field(..., examples=["PythonGetaway - MT5 Gateway"])
    version: str = Field(..., examples=["2.0.0"])
    mode: str = Field(..., examples=["multiusuario (pool de terminales)"])
    docs: str = Field(..., description="Ruta de la documentacion interactiva.", examples=["/docs"])
    openapi: str = Field(..., description="Ruta del esquema OpenAPI.", examples=["/openapi.json"])
    health: str = Field(..., examples=["/health"])
    login: str = Field(..., description="Por donde empezar.", examples=["POST /auth/login"])
    pool: dict[str, int] = Field(..., description="Recuento de slots por estado.",
                                 examples=[{"IDLE": 3}])
