"""Configuracion del gateway. Todo se controla por variables de entorno.

Ningun secreto vive en el codigo. Si no se define login/password/server, el
gateway se adjunta a la sesion que ya tenga abierta el terminal MT5 (que es el
caso normal: el terminal recuerda su cuenta)."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "si")


def _int_or_none(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else None


# --- Red -------------------------------------------------------------------
HOST = os.getenv("PYGW_HOST", "127.0.0.1")
PORT = int(os.getenv("PYGW_PORT", "8000"))

# --- Terminal MT5 ----------------------------------------------------------
# Ruta DENTRO del prefijo de Wine (formato Windows), no la ruta de Linux.
# Cada worker del pool apunta a SU PROPIA instancia portable; es lo que hace
# que las sesiones de distintos usuarios no se pisen.
TERMINAL_PATH = os.getenv("PYGW_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
MT5_LOGIN = _int_or_none("PYGW_LOGIN")
MT5_PASSWORD = os.getenv("PYGW_PASSWORD") or None
MT5_SERVER = os.getenv("PYGW_SERVER") or None
MT5_TIMEOUT_MS = int(os.getenv("PYGW_TIMEOUT_MS", "60000"))

# --- Modo worker (pool multiusuario) ---------------------------------------
# Con WORKER_MODE=1 el proceso NO se conecta a nada al arrancar: espera a que
# el gateway le mande unas credenciales por /internal/login. Es el modo que usa
# el pool. Sin el, el proceso se comporta como un gateway monousuario.
WORKER_MODE = _flag("PYGW_WORKER_MODE", False)
SLOT_ID = os.getenv("PYGW_SLOT_ID", "standalone")
# Secreto compartido gateway<->worker. Los workers escuchan en localhost, pero
# esto evita que cualquier proceso local pueda secuestrar un slot autenticado.
INTERNAL_TOKEN = os.getenv("PYGW_INTERNAL_TOKEN") or None

# --- Seguridad -------------------------------------------------------------
# Por defecto el gateway REHUSA operar sobre una cuenta REAL. Los endpoints de
# lectura siguen funcionando; solo se bloquea lo que envia ordenes.
# Para permitirlo explicitamente: PYGW_ALLOW_REAL=1
ALLOW_REAL_ACCOUNT = _flag("PYGW_ALLOW_REAL", False)

# Magic number por defecto de las ordenes que envie este gateway. Sirve para
# distinguir en MT5 lo que vino de aqui de lo que se hizo a mano en el terminal.
DEFAULT_MAGIC = int(os.getenv("PYGW_MAGIC", "20260910"))
DEFAULT_DEVIATION = int(os.getenv("PYGW_DEVIATION", "20"))

# --- Limites ---------------------------------------------------------------
MAX_BARS = int(os.getenv("PYGW_MAX_BARS", "50000"))
MAX_TICKS = int(os.getenv("PYGW_MAX_TICKS", "200000"))

RUN_DIR = BASE_DIR / "run"
LOG_DIR = BASE_DIR / "logs"
