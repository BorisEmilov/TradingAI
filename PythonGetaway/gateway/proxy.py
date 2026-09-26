"""Reenvio transparente de las peticiones al worker de cada sesion.

El gateway NO reimplementa los 43 endpoints: resuelve el token -> slot y
reenvia metodo, ruta, query y cuerpo al worker correspondiente. Asi el contrato
publico es exactamente el mismo que ya tenias, con la unica diferencia de que
ahora lleva `Authorization: Bearer`.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, Request, Response

from gateway import config
from gateway.auth import current_session
from gateway.errors import GatewayError

router = APIRouter()
log = logging.getLogger("pygw.proxy")

# Rutas que el gateway sirve por si mismo y NUNCA reenvia.
LOCAL_PREFIXES = ("/auth", "/docs", "/redoc", "/openapi.json", "/health", "/pool", "/favicon.ico")
# El worker expone /internal/* para hablar con el gateway. Exponerlo al exterior
# permitiria a cualquiera reautenticar un slot ajeno con otras credenciales.
BLOCKED_PREFIXES = ("/internal",)

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "upgrade",
              "proxy-authorization", "proxy-authenticate", "te", "trailer", "host",
              "content-length", "authorization"}


@router.api_route("/{full_path:path}",
                  methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
                  include_in_schema=False)
async def proxy(full_path: str, request: Request, session=Depends(current_session)) -> Response:
    path = "/" + full_path.lstrip("/")

    if any(path == p or path.startswith(p + "/") for p in BLOCKED_PREFIXES):
        raise GatewayError(404, "NOT_FOUND", "Ruta no disponible.")

    pool = request.app.state.pool
    slot = next((s for s in pool.slots if s.slot_id == session.slot_id), None)
    if slot is None or slot.state != "BUSY":
        raise GatewayError(
            409, "SESSION_SLOT_LOST",
            "El terminal de esta sesion ya no esta disponible (se reinicio o caduco). "
            "Vuelve a autenticarte en POST /auth/login.",
            detail={"slot_id": session.slot_id, "slot_state": None if slot is None else slot.state})

    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    headers["X-Internal-Token"] = config.INTERNAL_TOKEN

    url = f"{slot.base_url}{path}"
    try:
        upstream = await pool.client.request(
            request.method, url, params=dict(request.query_params),
            content=body or None, headers=headers, timeout=config.REQUEST_TIMEOUT)
    except httpx.TimeoutException as exc:
        raise GatewayError(504, "WORKER_TIMEOUT",
                           f"El worker {slot.slot_id} no respondio a tiempo.",
                           detail={"path": path}) from exc
    except Exception as exc:  # noqa: BLE001
        raise GatewayError(502, "WORKER_UNREACHABLE",
                           f"No se pudo hablar con el worker {slot.slot_id}: {exc}",
                           detail={"path": path}) from exc

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
    out_headers["X-Slot-Id"] = slot.slot_id
    out_headers["X-Account-Login"] = str(session.login)
    return Response(content=upstream.content, status_code=upstream.status_code,
                    headers=out_headers,
                    media_type=upstream.headers.get("content-type"))
