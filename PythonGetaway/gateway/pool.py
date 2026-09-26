"""Pool de workers MT5: un terminal + un proceso por cuenta concurrente.

POR QUE EXISTE ESTE MODULO
--------------------------
La libreria `MetaTrader5` es un SINGLETON DE PROCESO: `mt5.login()` cambia la
cuenta de todo el proceso y del terminal al que esta enganchado. Dos usuarios
en el mismo proceso comparten cuenta, asi que el "cerrar posicion" de uno
podria ejecutarse contra la cuenta del otro.

La unica forma de servir N cuentas a la vez es N terminales, cada uno con su
propio directorio de datos (modo `/portable`), y un proceso Python atado a cada
uno con `mt5.initialize(path=...)`. Eso es lo que gestiona este pool.

ESTADOS DE UN SLOT
------------------
    COLD     -> sin procesos
    WARMING  -> terminal y worker arrancando
    IDLE     -> listo y SIN cuenta: disponible para un login
    BUSY     -> autenticado con una cuenta concreta
    RECYCLING-> cerrando sesion y reiniciando el terminal
    FAILED   -> no se pudo levantar (ver `error`)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from gateway import config

log = logging.getLogger("pygw.pool")

COLD, WARMING, IDLE, BUSY, RECYCLING, FAILED = (
    "COLD", "WARMING", "IDLE", "BUSY", "RECYCLING", "FAILED")


@dataclass
class Slot:
    index: int
    slot_id: str
    port: int
    instance_dir: Path
    terminal_path_win: str
    state: str = COLD
    terminal_proc: subprocess.Popen | None = None
    worker_proc: subprocess.Popen | None = None
    account: dict | None = None
    error: str | None = None
    since: float = field(default_factory=time.time)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def public(self) -> dict:
        return {"slot_id": self.slot_id, "state": self.state, "port": self.port,
                "login": (self.account or {}).get("login"),
                "server": (self.account or {}).get("server"),
                "seconds_in_state": round(time.time() - self.since, 1),
                "error": self.error}


def _wine_env() -> dict:
    env = dict(os.environ)
    env.update({
        "WINEPREFIX": str(config.WINEPREFIX),
        "WINEARCH": "win64",
        "WINEDLLOVERRIDES": "mscoree=;mshtml=",
        "WINEDEBUG": "-all",          # sin esto Wine inunda el log
    })
    return env


class WorkerPool:
    def __init__(self, size: int = None):
        self.size = size or config.POOL_SIZE
        self.slots: list[Slot] = []
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT)
        for i in range(1, self.size + 1):
            slot_id = f"slot-{i}"
            self.slots.append(Slot(
                index=i, slot_id=slot_id, port=config.WORKER_PORT_BASE + i,
                instance_dir=config.INSTANCES_DIR / slot_id,
                terminal_path_win=f"{config.INSTANCES_DIR_WIN}\\{slot_id}\\terminal64.exe",
            ))

    # ---------------------------------------------------------------- ciclo
    async def start(self) -> None:
        config.INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Calentando pool de %d slots...", self.size)
        await asyncio.gather(*(self._warm(s) for s in self.slots), return_exceptions=True)
        ready = sum(1 for s in self.slots if s.state == IDLE)
        log.info("Pool listo: %d/%d slots disponibles.", ready, self.size)

    async def stop(self) -> None:
        log.info("Apagando pool...")
        await asyncio.gather(*(self._teardown(s) for s in self.slots), return_exceptions=True)
        await self._client.aclose()

    # ------------------------------------------------------------ provision
    def _provision(self, slot: Slot) -> None:
        """Clona la plantilla de MT5 en el directorio del slot (una sola vez).

        Sin `accounts.dat`: ese fichero lleva las credenciales guardadas de la
        cuenta del template y no debe acabar en el slot de otro usuario."""
        exe = slot.instance_dir / "terminal64.exe"
        if exe.exists():
            return
        if not (config.TEMPLATE_DIR / "terminal64.exe").exists():
            raise RuntimeError(f"No hay instalacion de MT5 en la plantilla: {config.TEMPLATE_DIR}")

        log.info("[%s] clonando instancia MT5 desde la plantilla...", slot.slot_id)
        excludes = set(config.TEMPLATE_EXCLUDES)

        def ignore(directory: str, names: list[str]) -> set[str]:
            rel = Path(directory).relative_to(config.TEMPLATE_DIR)
            skip = set()
            for name in names:
                candidate = str(rel / name).replace(os.sep, "/").lstrip("./")
                if name in excludes or candidate in excludes:
                    skip.add(name)
            return skip

        slot.instance_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(config.TEMPLATE_DIR, slot.instance_dir, ignore=ignore, dirs_exist_ok=True)
        # Defensa en profundidad por si la plantilla cambia de forma.
        for leftover in ("Config/accounts.dat", "Config/terminal.ini"):
            (slot.instance_dir / leftover).unlink(missing_ok=True)
        log.info("[%s] instancia lista en %s", slot.slot_id, slot.instance_dir)

    async def _warm(self, slot: Slot) -> None:
        """Deja el slot en IDLE: terminal en marcha y worker escuchando, sin cuenta."""
        slot.state, slot.since, slot.error = WARMING, time.time(), None
        try:
            await asyncio.to_thread(self._provision, slot)
            await asyncio.to_thread(self._launch_terminal, slot)
            await asyncio.sleep(config.TERMINAL_BOOT_WAIT)
            await asyncio.to_thread(self._launch_worker, slot)
            await self._wait_worker(slot)
            slot.state, slot.since, slot.account = IDLE, time.time(), None
            log.info("[%s] IDLE en el puerto %d", slot.slot_id, slot.port)
        except Exception as exc:  # noqa: BLE001
            slot.state, slot.error, slot.since = FAILED, str(exc), time.time()
            log.error("[%s] no se pudo calentar: %s", slot.slot_id, exc)

    def _launch_terminal(self, slot: Slot) -> None:
        """Arranca el terminal en modo portable.

        Se lanza aparte (en vez de dejar que `mt5.initialize()` lo haga) para
        que el arranque de ~15 s ocurra al calentar el pool y no dentro del
        login del usuario."""
        if slot.terminal_proc and slot.terminal_proc.poll() is None:
            return
        exe = slot.instance_dir / "terminal64.exe"
        logfile = open(config.LOG_DIR / f"{slot.slot_id}-terminal.log", "ab")
        slot.terminal_proc = subprocess.Popen(
            ["wine", str(exe), "/portable"], env=_wine_env(),
            stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
        log.info("[%s] terminal lanzado (pid %s)", slot.slot_id, slot.terminal_proc.pid)

    def _launch_worker(self, slot: Slot) -> None:
        """Arranca el proceso worker (FastAPI bajo Wine) atado a ESTE terminal."""
        if slot.worker_proc and slot.worker_proc.poll() is None:
            return
        env = _wine_env()
        env.update({
            "PYGW_WORKER_MODE": "1",
            "PYGW_SLOT_ID": slot.slot_id,
            "PYGW_TERMINAL_PATH": slot.terminal_path_win,
            "PYGW_INTERNAL_TOKEN": config.INTERNAL_TOKEN,
            "PYGW_ALLOW_REAL": "1" if config.ALLOW_REAL_ACCOUNT else "0",
        })
        # `script` da una pseudo-tty: el python.exe de Wine aborta con
        # "init_sys_streams ... Invalid handle" si no la tiene.
        inner = " ".join([
            "wine", _q(str(config.WINE_PYTHON)), "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(slot.port),
            "--workers", "1", "--log-level", "warning",
        ])
        logfile = open(config.LOG_DIR / f"{slot.slot_id}-worker.log", "ab")
        slot.worker_proc = subprocess.Popen(
            ["script", "-qec", inner, "/dev/null"],
            cwd=str(config.BASE_DIR), env=env,
            stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
        log.info("[%s] worker lanzado (pid %s, puerto %d)",
                 slot.slot_id, slot.worker_proc.pid, slot.port)

    async def _wait_worker(self, slot: Slot) -> None:
        deadline = time.time() + config.WORKER_BOOT_TIMEOUT
        while time.time() < deadline:
            try:
                r = await self._client.get(f"{slot.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except Exception:  # noqa: BLE001 -- aun no escucha
                pass
            if slot.worker_proc and slot.worker_proc.poll() is not None:
                raise RuntimeError(f"el worker murio al arrancar (ver logs/{slot.slot_id}-worker.log)")
            await asyncio.sleep(1)
        raise RuntimeError(f"el worker no respondio en {config.WORKER_BOOT_TIMEOUT}s")

    async def _teardown(self, slot: Slot, keep_terminal: bool = False) -> None:
        for proc, label in ((slot.worker_proc, "worker"),
                            (None if keep_terminal else slot.terminal_proc, "terminal")):
            if proc is None or proc.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:  # noqa: BLE001
                proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 15)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    proc.kill()
            log.info("[%s] %s detenido", slot.slot_id, label)
        slot.worker_proc = None
        if not keep_terminal:
            slot.terminal_proc = None
        slot.state, slot.since, slot.account = COLD, time.time(), None

    # --------------------------------------------------------------- alquiler
    async def acquire(self, login: int, password: str, server: str) -> Slot:
        """Reserva un slot y lo autentica. Lanza si no hay ninguno libre."""
        async with self._lock:
            slot = next((s for s in self.slots if s.state == IDLE), None)
            if slot is None:
                raise PoolExhausted(
                    f"No hay slots libres (pool de {self.size}). "
                    "Cierra alguna sesion o aumenta PYGW_POOL_SIZE.")
            slot.state, slot.since = BUSY, time.time()   # reservado antes de soltar el lock

        try:
            r = await self._client.post(
                f"{slot.base_url}/internal/login",
                json={"login": login, "password": password, "server": server},
                headers={"X-Internal-Token": config.INTERNAL_TOKEN},
                timeout=config.LOGIN_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            slot.state, slot.since = IDLE, time.time()
            raise WorkerUnreachable(f"El worker {slot.slot_id} no respondio: {exc}") from exc

        if r.status_code != 200:
            slot.state, slot.since = IDLE, time.time()
            raise LoginFailed(r.status_code, _payload(r))

        data = r.json()
        slot.account = data.get("account") or {}
        slot.since = time.time()
        return slot

    async def release(self, slot: Slot) -> None:
        """Cierra la sesion del slot y lo devuelve al pool."""
        if slot.state not in (BUSY,):
            return
        slot.state, slot.since = RECYCLING, time.time()
        try:
            await self._client.post(
                f"{slot.base_url}/internal/logout",
                headers={"X-Internal-Token": config.INTERNAL_TOKEN}, timeout=30)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] fallo al cerrar sesion en el worker: %s", slot.slot_id, exc)
        slot.account = None

        if config.RECYCLE_ON_LOGOUT:
            # Reiniciar el terminal para que NINGUNA cuenta quede conectada.
            # Se hace en segundo plano: el logout del usuario responde ya.
            asyncio.create_task(self._recycle(slot))
        else:
            slot.state, slot.since = IDLE, time.time()

    async def _recycle(self, slot: Slot) -> None:
        try:
            await self._teardown(slot)
            await self._warm(slot)
        except Exception as exc:  # noqa: BLE001
            slot.state, slot.error = FAILED, str(exc)
            log.error("[%s] fallo al reciclar: %s", slot.slot_id, exc)

    # ---------------------------------------------------------------- estado
    def status(self) -> dict:
        counts: dict[str, int] = {}
        for s in self.slots:
            counts[s.state] = counts.get(s.state, 0) + 1
        return {"size": self.size, "states": counts,
                "available": sum(1 for s in self.slots if s.state == IDLE),
                "slots": [s.public() for s in self.slots]}

    async def worker_openapi(self) -> dict | None:
        """Esquema OpenAPI de cualquier worker vivo (para componer el publico)."""
        for slot in self.slots:
            if slot.state in (IDLE, BUSY):
                try:
                    r = await self._client.get(f"{slot.base_url}/openapi.json", timeout=10)
                    if r.status_code == 200:
                        return r.json()
                except Exception:  # noqa: BLE001
                    continue
        return None

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client


def _q(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def _payload(response: httpx.Response) -> dict:
    try:
        return response.json()
    except Exception:  # noqa: BLE001
        return {"raw": response.text[:500]}


class PoolExhausted(RuntimeError):
    pass


class WorkerUnreachable(RuntimeError):
    pass


class LoginFailed(RuntimeError):
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"login rechazado ({status_code})")
