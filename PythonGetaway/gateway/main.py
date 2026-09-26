"""PythonGetaway -- gateway multiusuario sobre MetaTrader 5.

Corre en el Python de LINUX (no bajo Wine): es la puerta de entrada, atiende a
todos los clientes y solo enruta, asi que no necesita la libreria MT5 y gana
concurrencia real. Los workers, que si la necesitan, viven bajo Wine.

    Cliente .NET
        |  HTTP + Authorization: Bearer <token>
        v
    GATEWAY (Linux, async)  --- /auth/*, sesiones, pool
        |  reenvio interno por localhost
        v
    WORKER slot-1 (Wine) --> terminal MT5 propio --> cuenta del usuario A
    WORKER slot-2 (Wine) --> terminal MT5 propio --> cuenta del usuario B
    ...
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway import auth, config, proxy
from gateway.errors import GatewayError
from gateway.models import GatewayIndexResponse, HealthResponse, PoolStatus
from gateway.pool import WorkerPool
from gateway.sessions import SessionStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("pygw.gateway")



async def _snapshot_worker_schema(app: FastAPI) -> None:
    """Cachea el esquema OpenAPI de un worker vivo.

    Se guarda tambien en disco para poder publicar el contrato completo aunque
    en un arranque posterior ningun worker llegue a levantarse."""
    try:
        schema = await app.state.pool.worker_openapi()
        if schema:
            app.state.worker_schema = schema
            config.WORKER_SCHEMA_SNAPSHOT.write_text(json.dumps(schema))
            log.info("Esquema del worker cacheado (%d rutas).", len(schema.get("paths", {})))
        else:
            log.warning("Ningun worker devolvio su esquema OpenAPI.")
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo cachear el esquema del worker: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)

    app.state.pool = WorkerPool()
    app.state.sessions = SessionStore(app.state.pool)
    app.state.worker_schema = None

    await app.state.pool.start()
    await app.state.sessions.start_sweeper()
    await _snapshot_worker_schema(app)
    log.info("Gateway escuchando en %s:%s (pool de %d slots)",
             config.HOST, config.PORT, config.POOL_SIZE)
    yield
    await app.state.sessions.stop_sweeper()
    await app.state.pool.stop()


app = FastAPI(
    title="PythonGetaway - MT5 Gateway (multiusuario)",
    version="2.0.0",
    description=(
        "API HTTP sobre MetaTrader 5 con **autenticacion por usuario**.\n\n"
        "Cada usuario se autentica con SUS credenciales MT5 en `POST /auth/login` y recibe "
        "un token. Todo lo demas se llama con `Authorization: Bearer <token>`.\n\n"
        "**Aislamiento**: la libreria MT5 es un singleton de proceso, asi que cada sesion "
        "recibe su **propio terminal MT5 y su propio proceso**. Dos usuarios nunca comparten "
        "cuenta.\n\n"
        "**Credenciales**: no se guardan en ningun sitio. Se usan una vez para autenticar el "
        "terminal y se descartan. Si el gateway se reinicia, hay que volver a autenticarse.\n\n"
        "**Tiempos**: los epochs son hora del SERVIDOR del broker, no UTC. Cada campo sale como "
        "`xxx` (epoch) y `xxx_iso` (ISO-8601 sin sufijo de zona).\n\n"
        "**Ordenes**: un HTTP 200 con `success: false` significa que MT5 recibio la peticion y la "
        "rechazo. El motivo esta en `retcode_name`."
    ),
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("Error no controlado en %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": {
        "code": "INTERNAL_ERROR", "message": str(exc), "mt5_code": None,
        "detail": {"path": request.url.path}}})


@app.get("/", tags=["gateway"], summary="Indice del gateway",
         response_model=GatewayIndexResponse)
async def root(request: Request) -> dict:
    return {
        "name": "PythonGetaway - MT5 Gateway",
        "version": "2.0.0",
        "mode": "multiusuario (pool de terminales)",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "login": "POST /auth/login",
        "pool": request.app.state.pool.status()["states"],
    }


@app.get("/health", tags=["gateway"], summary="Estado del gateway y del pool",
         response_model=HealthResponse)
async def health(request: Request) -> dict:
    """No requiere token: es el endpoint de sonda.

    `available` es cuantos usuarios nuevos pueden autenticarse ahora mismo."""
    pool = request.app.state.pool
    status = pool.status()
    return {
        "status": "ok" if status["available"] > 0 else "saturado",
        "pool": status,
        "sessions_active": len(request.app.state.sessions.all()),
        "allow_real_account": config.ALLOW_REAL_ACCOUNT,
        "recycle_on_logout": config.RECYCLE_ON_LOGOUT,
        "session_ttl_seconds": config.SESSION_TTL,
        "session_idle_timeout_seconds": config.SESSION_IDLE_TIMEOUT,
    }


@app.get("/pool", tags=["gateway"], summary="Detalle del pool de terminales",
         response_model=PoolStatus)
async def pool_status(request: Request) -> dict:
    """Estado de cada slot. Util para operar y dimensionar."""
    return request.app.state.pool.status()


app.include_router(auth.router)


# ---------------------------------------------------------------------------
# OpenAPI compuesto: /auth/* (del gateway) + los 43 endpoints (de un worker).
# ---------------------------------------------------------------------------
_BASE_OPENAPI = app.openapi


def _compose_openapi() -> dict:
    schema = _BASE_OPENAPI()
    worker = getattr(app.state, "worker_schema", None)
    if worker is None and config.WORKER_SCHEMA_SNAPSHOT.exists():
        try:
            worker = json.loads(config.WORKER_SCHEMA_SNAPSHOT.read_text())
        except Exception:  # noqa: BLE001
            worker = None
    if not worker:
        return schema

    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})["bearerAuth"] = {
        "type": "http", "scheme": "bearer",
        "description": "Token devuelto por POST /auth/login.",
    }

    # Los endpoints propios que exigen token lo reciben via `Header(None)`, asi
    # que FastAPI los publica como un parametro `authorization` OPCIONAL: falso
    # (el token es obligatorio) y ademas ensucia el cliente generado. Se
    # sustituye por el esquema de seguridad, que es la forma correcta.
    for path in ("/auth/logout", "/auth/session", "/auth/refresh"):
        for op in (schema.get("paths", {}).get(path, {}) or {}).values():
            op["security"] = [{"bearerAuth": []}]
            op["parameters"] = [prm for prm in op.get("parameters", [])
                                if prm.get("name", "").lower() != "authorization"]
    for name, definition in (worker.get("components", {}).get("schemas", {}) or {}).items():
        components.setdefault("schemas", {}).setdefault(name, definition)

    for path, methods in (worker.get("paths", {}) or {}).items():
        if path.startswith("/internal") or path in ("/", "/health"):
            continue
        entry = {}
        for method, op in methods.items():
            op = dict(op)
            op["security"] = [{"bearerAuth": []}]
            entry[method] = op
        schema["paths"].setdefault(path, entry)
    return schema


app.openapi = _compose_openapi

# Nota: `@app.on_event("startup")` NO se ejecuta cuando la app declara
# `lifespan` -- FastAPI ignora los handlers antiguos en ese caso. Por eso
# `_snapshot_worker_schema()` se invoca explicitamente desde el lifespan.

# El proxy va el ULTIMO: su ruta comodin /{full_path} capturaria cualquier cosa
# declarada despues.
app.include_router(proxy.router)
