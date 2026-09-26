"""Sesion MT5: ciclo de vida de la conexion y acceso serializado.

DOS COSAS IMPORTANTES SOBRE LA LIBRERIA MetaTrader5:

1. **No es thread-safe.** Es un wrapper sobre una DLL con estado global de
   proceso. Dos peticiones HTTP concurrentes que llamen a mt5.* a la vez pueden
   devolver datos cruzados o corromper el estado. Por eso TODA llamada pasa por
   `MT5_LOCK`, y los endpoints se declaran `def` (no `async def`) para que
   FastAPI los ejecute en su threadpool sin bloquear el event loop.

2. **Es Windows-only.** Envuelve la DLL del terminal, que es una aplicacion
   Windows. En Linux este proceso corre bajo Wine (ver scripts/start.sh).
"""

from __future__ import annotations

import threading
from typing import Any

import MetaTrader5 as mt5

from app import config
from app.errors import MT5Error, NotConnectedError, RealAccountBlockedError

# Un unico lock global para todas las llamadas a la libreria (ver docstring).
MT5_LOCK = threading.RLock()

# account_info().trade_mode
TRADE_MODE_DEMO = 0
TRADE_MODE_CONTEST = 1
TRADE_MODE_REAL = 2
TRADE_MODE_NAMES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}

_state: dict[str, Any] = {"initialized": False, "last_error": None}


# ---------------------------------------------------------------------------
# Ciclo de vida
# ---------------------------------------------------------------------------
def initialize(login: int | None = None, password: str | None = None,
               server: str | None = None) -> dict:
    """Conecta con el terminal MT5 de ESTE worker (`config.TERMINAL_PATH`).

    `path` no es opcional en el pool: es lo que ata el proceso a su propia
    instancia portable. Sin el, la libreria se engancharia a cualquier terminal
    en marcha y dos usuarios acabarian compartiendo cuenta.

    Las credenciales, si llegan, se usan y se descartan: no se guardan en
    ningun sitio (ni aqui, ni en disco, ni en logs)."""
    with MT5_LOCK:
        kwargs: dict[str, Any] = {"timeout": config.MT5_TIMEOUT_MS}
        if config.TERMINAL_PATH:
            kwargs["path"] = config.TERMINAL_PATH

        login = login if login is not None else config.MT5_LOGIN
        password = password if password is not None else config.MT5_PASSWORD
        server = server if server is not None else config.MT5_SERVER
        if login and password and server:
            kwargs.update(login=int(login), password=password, server=server)

        if not mt5.initialize(**kwargs):
            code, desc = mt5.last_error()
            _state["initialized"] = False
            _state["last_error"] = {"code": code, "description": desc}
            raise MT5Error(f"No se pudo inicializar el terminal MT5: {desc}", mt5_code=code)

        _state["initialized"] = True
        _state["last_error"] = None

        # Verificacion explicita: `initialize()` puede devolver True habiendose
        # conectado al terminal pero SIN haber autenticado la cuenta pedida.
        # Dar por buena la sesion en ese caso serviria al usuario los datos de
        # quien estuviera logueado antes en este slot.
        if login:
            info = mt5.account_info()
            if info is None or int(info.login) != int(login):
                actual = None if info is None else int(info.login)
                mt5.shutdown()
                _state["initialized"] = False
                code, desc = mt5.last_error()
                raise MT5Error(
                    "El terminal no quedo autenticado con la cuenta solicitada.",
                    mt5_code=code,
                    detail={"requested_login": int(login), "actual_login": actual,
                            "mt5_description": desc})
        return status()


def login_account(login: int, password: str, server: str) -> dict:
    """Autentica este worker contra una cuenta concreta (modo pool).

    Si el worker ya tenia sesion, se cierra antes: un slot nunca sirve a dos
    cuentas a la vez."""
    with MT5_LOCK:
        if _state.get("initialized"):
            mt5.shutdown()
            _state["initialized"] = False
        return initialize(login=login, password=password, server=server)


def logout_account() -> dict:
    """Cierra la sesion del worker y lo deja libre para otro usuario."""
    with MT5_LOCK:
        shutdown()
        return {"logged_out": True, "slot_id": config.SLOT_ID}


def shutdown() -> None:
    with MT5_LOCK:
        if _state.get("initialized"):
            mt5.shutdown()
        _state["initialized"] = False


def status() -> dict:
    """Estado de la conexion. Nunca lanza: sirve para el health check."""
    with MT5_LOCK:
        if not _state.get("initialized"):
            return {"connected": False, "initialized": False, "account": None,
                    "terminal": None, "last_error": _state.get("last_error")}
        ai = mt5.account_info()
        ti = mt5.terminal_info()
        version = mt5.version()
        return {
            "connected": bool(ti.connected) if ti else False,
            "initialized": True,
            "account": {
                "login": ai.login, "server": ai.server, "currency": ai.currency,
                "balance": ai.balance, "equity": ai.equity,
                "trade_mode": ai.trade_mode,
                "trade_mode_name": TRADE_MODE_NAMES.get(ai.trade_mode, "UNKNOWN"),
                "trade_allowed": bool(ai.trade_allowed),
            } if ai else None,
            "terminal": {
                "build": version[1] if version else None,
                "version": version[0] if version else None,
                "path": ti.path if ti else None,
                "trade_allowed": bool(ti.trade_allowed) if ti else False,
                "connected": bool(ti.connected) if ti else False,
            },
            "trading_enabled_by_gateway": _trading_allowed()[0],
            "last_error": None,
        }


def require_connection() -> None:
    with MT5_LOCK:
        if not _state.get("initialized"):
            raise NotConnectedError()
        ti = mt5.terminal_info()
        if ti is None or not ti.connected:
            raise NotConnectedError(
                "El terminal MT5 esta abierto pero no tiene conexion con el servidor del broker.")


# ---------------------------------------------------------------------------
# Guardarrail de cuenta real
# ---------------------------------------------------------------------------
def _trading_allowed() -> tuple[bool, int | None, int | None]:
    """(permitido, trade_mode, login). Nunca lanza."""
    if not _state.get("initialized"):
        return False, None, None
    ai = mt5.account_info()
    if ai is None:
        return False, None, None
    if ai.trade_mode == TRADE_MODE_REAL and not config.ALLOW_REAL_ACCOUNT:
        return False, ai.trade_mode, ai.login
    return True, ai.trade_mode, ai.login


def require_trading_allowed() -> None:
    """Se invoca en TODOS los endpoints que envian ordenes."""
    require_connection()
    with MT5_LOCK:
        ok, trade_mode, login = _trading_allowed()
        if not ok and trade_mode == TRADE_MODE_REAL:
            raise RealAccountBlockedError(trade_mode, login)


# ---------------------------------------------------------------------------
# Utilidades de simbolo
# ---------------------------------------------------------------------------
def ensure_symbol(symbol: str):
    """Devuelve el `symbol_info`, seleccionandolo en Market Watch si hace falta.

    Un simbolo no seleccionado no devuelve cotizaciones ni velas, asi que este
    paso es obligatorio antes de cualquier lectura de mercado."""
    from app.errors import NotFoundError
    with MT5_LOCK:
        info = mt5.symbol_info(symbol)
        if info is None:
            code, desc = mt5.last_error()
            raise NotFoundError(f"Simbolo desconocido para el broker: {symbol}",
                                detail={"mt5_code": code, "mt5_description": desc})
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                code, desc = mt5.last_error()
                raise MT5Error(f"No se pudo seleccionar el simbolo {symbol}: {desc}", mt5_code=code)
            info = mt5.symbol_info(symbol)
        return info


def pick_filling_mode(symbol: str) -> int:
    """Modo de llenado aceptado por el simbolo.

    `symbol_info().filling_mode` es una MASCARA DE BITS de lo que admite el
    broker (bit 1 = FOK, bit 2 = IOC). Los nombres SYMBOL_FILLING_* no estan
    expuestos en este binding de Python, asi que se usan los valores de bit
    documentados en MQL5. Enviar un modo no admitido devuelve el retcode 10030
    ("Unsupported filling mode"), que es la causa mas comun de que una orden
    correcta sea rechazada."""
    info = ensure_symbol(symbol)
    mask = int(getattr(info, "filling_mode", 0) or 0)
    if mask & 1:
        return mt5.ORDER_FILLING_FOK
    if mask & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def normalize_volume(symbol: str, volume: float) -> float:
    """Ajusta el volumen al paso del simbolo y lo valida contra min/max.

    Se redondea SIEMPRE hacia abajo al `volume_step`: redondear hacia arriba
    aumentaria el riesgo pedido por el cliente sin que este lo sepa."""
    from app.errors import BadRequestError
    info = ensure_symbol(symbol)
    step = float(info.volume_step or 0.01)
    vmin, vmax = float(info.volume_min), float(info.volume_max)
    if volume <= 0:
        raise BadRequestError("El volumen debe ser mayor que 0.", detail={"volume": volume})
    steps = round(volume / step)
    snapped = round(steps * step, 8)
    if abs(snapped - volume) > 1e-9:
        # se ajusta hacia abajo, nunca hacia arriba
        snapped = round((int(volume / step + 1e-9)) * step, 8)
    if snapped < vmin:
        raise BadRequestError(
            f"Volumen {volume} por debajo del minimo del simbolo ({vmin}).",
            detail={"symbol": symbol, "volume_min": vmin, "volume_step": step})
    if snapped > vmax:
        raise BadRequestError(
            f"Volumen {volume} por encima del maximo del simbolo ({vmax}).",
            detail={"symbol": symbol, "volume_max": vmax})
    return snapped


def last_error() -> dict:
    with MT5_LOCK:
        code, desc = mt5.last_error()
        return {"mt5_code": code, "detail": desc}
