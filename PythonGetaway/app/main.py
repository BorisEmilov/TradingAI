"""PythonGetaway -- gateway HTTP sobre MetaTrader 5.

Corre bajo el Python de Windows dentro de Wine (ver scripts/start.sh), porque
la libreria `MetaTrader5` envuelve la DLL del terminal, que es una aplicacion
Windows y no existe de forma nativa en Linux.

Documentacion interactiva una vez arrancado:
    http://127.0.0.1:8000/docs        (Swagger UI -- se puede probar desde ahi)
    http://127.0.0.1:8000/redoc       (ReDoc)
    http://127.0.0.1:8000/openapi.json (esquema para generar el cliente .NET)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import config, mt5_session
from app.errors import GatewayError
from app.routers import (account, history, internal, market, orders, positions,
                         symbols, trading)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("pygw")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.WORKER_MODE:
        # En el pool, un worker arranca VACIO y espera a que el gateway le pase
        # credenciales. Conectarse aqui con la sesion que quedara guardada en el
        # terminal serviria datos de otra persona al primer usuario que caiga
        # en este slot.
        log.info("Worker %s listo (sin sesion). Esperando /internal/login.", config.SLOT_ID)
        yield
        log.info("Cerrando worker %s...", config.SLOT_ID)
        mt5_session.shutdown()
        return

    log.info("Conectando con el terminal MT5...")
    try:
        status = mt5_session.initialize()
        acc = status.get("account") or {}
        log.info("Conectado: login=%s server=%s tipo=%s balance=%s %s",
                 acc.get("login"), acc.get("server"), acc.get("trade_mode_name"),
                 acc.get("balance"), acc.get("currency"))
        if acc.get("trade_mode_name") == "REAL" and not config.ALLOW_REAL_ACCOUNT:
            log.warning("CUENTA REAL detectada y PYGW_ALLOW_REAL no esta activo: "
                        "los endpoints de trading responderan 403. La lectura funciona.")
    except Exception as exc:  # noqa: BLE001 -- el gateway arranca igual
        # Arrancar aunque MT5 no responda permite que /health explique QUE pasa
        # en vez de dejar al cliente con "connection refused".
        log.error("No se pudo conectar con MT5 al arrancar: %s", exc)
    yield
    log.info("Cerrando conexion con MT5...")
    mt5_session.shutdown()


app = FastAPI(
    title="PythonGetaway - MT5 Gateway",
    version="1.0.0",
    description=(
        "API HTTP sobre MetaTrader 5: extraccion de datos y trading manual.\n\n"
        "**Tiempos**: MT5 devuelve epochs medidos en la hora del SERVIDOR del broker. "
        "Cada campo temporal se expone como `xxx` (epoch entero) y `xxx_iso` "
        "(ISO-8601 sin sufijo de zona, porque no es UTC sino hora de servidor).\n\n"
        "**Errores**: todos comparten la forma "
        "`{\"error\": {\"code\": ..., \"message\": ..., \"mt5_code\": ..., \"detail\": ...}}`.\n\n"
        "**Ordenes**: un HTTP 200 con `success: false` significa que MT5 recibio la "
        "peticion y la rechazo. El motivo esta en `retcode_name`."
    ),
    lifespan=lifespan,
)

# El cliente es .NET; si algun dia se llama desde un navegador, esto ya esta.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """Ningun fallo inesperado debe llegar al cliente como HTML de traceback."""
    log.exception("Error no controlado en %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": {
        "code": "INTERNAL_ERROR", "message": str(exc), "mt5_code": None,
        "detail": {"path": request.url.path},
    }})


@app.get("/", tags=["account"], summary="Indice del gateway")
def root() -> dict:
    return {
        "name": "PythonGetaway - MT5 Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        # Se cuenta desde el esquema OpenAPI, no desde `app.routes`: esta version
        # de FastAPI guarda los routers incluidos como objetos `_IncludedRouter`
        # sin aplanar, asi que recorrer `app.routes` deja fuera casi todo.
        "endpoint_count": len(app.openapi().get("paths", {})),
    }


app.include_router(account.router)
app.include_router(symbols.router)
app.include_router(market.router)
app.include_router(positions.router)
app.include_router(orders.router)
app.include_router(history.router)
app.include_router(trading.router)
app.include_router(internal.router)
