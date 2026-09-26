"""Configuracion del gateway multiusuario."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HOME = Path(os.path.expanduser("~"))


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on", "si")


# --- Red publica -----------------------------------------------------------
HOST = os.getenv("PYGW_HOST", "127.0.0.1")
PORT = int(os.getenv("PYGW_PORT", "8000"))

# --- Wine / MT5 ------------------------------------------------------------
WINEPREFIX = Path(os.getenv("WINEPREFIX", str(HOME / ".wine-mt5")))
DRIVE_C = WINEPREFIX / "drive_c"
WINE_PYTHON = Path(os.getenv(
    "PYGW_WINE_PYTHON",
    str(DRIVE_C / "users" / os.getenv("USER", "user") / "AppData" / "Local" /
        "Programs" / "Python" / "Python311" / "python.exe")))

# Instalacion de MT5 que sirve de PLANTILLA para clonar cada slot.
TEMPLATE_DIR = Path(os.getenv("PYGW_MT5_TEMPLATE", str(DRIVE_C / "Program Files" / "MetaTrader 5")))
# Donde viven las instancias por slot (una por usuario concurrente).
INSTANCES_DIR = Path(os.getenv("PYGW_INSTANCES_DIR", str(DRIVE_C / "mt5-instances")))
INSTANCES_DIR_WIN = os.getenv("PYGW_INSTANCES_DIR_WIN", r"C:\mt5-instances")

# Ficheros de la plantilla que NUNCA se copian a un slot:
#   Bases/Tester/logs -> caches enormes, cada slot genera los suyos
#   accounts.dat      -> CREDENCIALES GUARDADAS de la cuenta del template.
#                        Copiarlas dejaria la sesion de otra persona accesible
#                        al primer usuario que aterrice en ese slot.
#   terminal.ini      -> recuerda la ultima cuenta usada
# `servers.dat` SI se copia: sin el, el terminal no sabe resolver el nombre
# del servidor del broker y el login por nombre falla.
TEMPLATE_EXCLUDES = ("Bases", "Tester", "logs", "MQL5/Files", "MQL5/Logs",
                     "Config/accounts.dat", "Config/terminal.ini")

# --- Pool ------------------------------------------------------------------
POOL_SIZE = int(os.getenv("PYGW_POOL_SIZE", "3"))
WORKER_PORT_BASE = int(os.getenv("PYGW_WORKER_PORT_BASE", "8100"))
WORKER_BOOT_TIMEOUT = int(os.getenv("PYGW_WORKER_BOOT_TIMEOUT", "120"))
TERMINAL_BOOT_WAIT = int(os.getenv("PYGW_TERMINAL_BOOT_WAIT", "12"))
LOGIN_TIMEOUT = int(os.getenv("PYGW_LOGIN_TIMEOUT", "90"))
REQUEST_TIMEOUT = int(os.getenv("PYGW_REQUEST_TIMEOUT", "120"))

# Al cerrar sesion, reiniciar el terminal del slot para que no quede ninguna
# cuenta conectada. Cuesta ~20 s de re-calentamiento, pero es lo correcto en
# multiusuario. Con 0 el slot se reutiliza al instante (mas rapido, y el
# terminal sigue logueado en la ultima cuenta hasta el siguiente login).
RECYCLE_ON_LOGOUT = _flag("PYGW_RECYCLE_ON_LOGOUT", True)

# --- Sesiones --------------------------------------------------------------
# Politica: el terminal vive MIENTRAS HAYA ACTIVIDAD. Si el cliente no hace
# ninguna peticion autenticada durante `SESSION_IDLE_TIMEOUT`, el terminal se
# cierra solo y el usuario tiene que volver a abrirlo con sus credenciales.
SESSION_IDLE_TIMEOUT = int(os.getenv("PYGW_SESSION_IDLE", "7200"))   # 2 h sin actividad

# Tope absoluto de vida de un token, ACTIVIDAD APARTE. Desactivado (0) por
# defecto: un tope duro echaria a un cliente que esta trabajando, que es justo
# lo contrario de la politica de arriba. Ponlo a un valor > 0 si quieres acotar
# cuanto tiempo puede circular un token pase lo que pase (se renueva con
# `POST /auth/refresh`).
SESSION_TTL = int(os.getenv("PYGW_SESSION_TTL", "0"))
SWEEP_INTERVAL = int(os.getenv("PYGW_SWEEP_INTERVAL", "30"))

# --- Seguridad -------------------------------------------------------------
# Secreto compartido gateway<->workers. Se genera en cada arranque si no viene
# dado: los workers solo escuchan en localhost y lo reciben por entorno.
INTERNAL_TOKEN = os.getenv("PYGW_INTERNAL_TOKEN") or secrets.token_urlsafe(32)
ALLOW_REAL_ACCOUNT = _flag("PYGW_ALLOW_REAL", False)
MAX_LOGIN_ATTEMPTS = int(os.getenv("PYGW_MAX_LOGIN_ATTEMPTS", "5"))
LOGIN_ATTEMPT_WINDOW = int(os.getenv("PYGW_LOGIN_ATTEMPT_WINDOW", "300"))

RUN_DIR = BASE_DIR / "run"
LOG_DIR = BASE_DIR / "logs"
WORKER_SCHEMA_SNAPSHOT = BASE_DIR / "gateway" / "worker_openapi.json"
