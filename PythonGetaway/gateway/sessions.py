"""Sesiones de usuario. **Todo en memoria, nada en disco.**

Las credenciales MT5 no se guardan aqui: viajan una sola vez del cliente al
worker en `/auth/login`, el worker autentica su terminal y se descartan. Lo
unico que persiste mientras dura la sesion es el token y a que slot apunta.

Consecuencia asumida: si el gateway se reinicia, todos los usuarios vuelven a
autenticarse. Es el precio de no ser un almacen de credenciales.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field

from gateway import config

log = logging.getLogger("pygw.sessions")


@dataclass
class Session:
    token: str
    slot_id: str
    login: int
    server: str
    account: dict
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def expires_at(self) -> float:
        """Vence por vida maxima o por inactividad, lo que llegue antes.

        `SESSION_TTL = 0` desactiva el tope absoluto: la sesion vive mientras el
        cliente siga trabajando y solo muere por inactividad."""
        idle_deadline = self.last_seen + config.SESSION_IDLE_TIMEOUT
        if config.SESSION_TTL <= 0:
            return idle_deadline
        return min(self.created_at + config.SESSION_TTL, idle_deadline)

    def renew(self) -> None:
        """Reinicia el reloj de vida maxima (`POST /auth/refresh`).

        El tope absoluto existe para acotar cuanto tiempo puede circular un
        token, no para echar a un cliente que esta trabajando. Sin esto, una
        sesion activa moria a la hora y obligaba a reenviar las credenciales
        -- justo lo que el token trata de evitar."""
        now = time.time()
        self.created_at = now
        self.last_seen = now

    def is_expired(self, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at()

    def public(self, include_token: bool = False) -> dict:
        data = {
            "slot_id": self.slot_id,
            "login": self.login,
            "server": self.server,
            "account": self.account,
            "created_at": int(self.created_at),
            "last_seen": int(self.last_seen),
            "expires_at": int(self.expires_at()),
            "expires_in": max(0, int(self.expires_at() - time.time())),
            "ttl_seconds": config.SESSION_TTL,
            "idle_timeout_seconds": config.SESSION_IDLE_TIMEOUT,
        }
        if include_token:
            data["token"] = self.token
        return data


class SessionStore:
    def __init__(self, pool):
        self._pool = pool
        self._by_token: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._sweeper: asyncio.Task | None = None
        # login -> [timestamps] para frenar fuerza bruta de credenciales
        self._attempts: dict[str, list[float]] = {}

    # ------------------------------------------------------------ anti-abuso
    def note_attempt(self, key: str) -> None:
        now = time.time()
        window = self._attempts.setdefault(key, [])
        window.append(now)
        self._attempts[key] = [t for t in window if now - t < config.LOGIN_ATTEMPT_WINDOW]

    def too_many_attempts(self, key: str) -> bool:
        now = time.time()
        window = [t for t in self._attempts.get(key, []) if now - t < config.LOGIN_ATTEMPT_WINDOW]
        self._attempts[key] = window
        return len(window) >= config.MAX_LOGIN_ATTEMPTS

    def clear_attempts(self, key: str) -> None:
        self._attempts.pop(key, None)

    # --------------------------------------------------------------- CRUD
    async def create(self, slot, login: int, server: str) -> Session:
        async with self._lock:
            session = Session(token=secrets.token_urlsafe(32), slot_id=slot.slot_id,
                              login=login, server=server, account=slot.account or {})
            self._by_token[session.token] = session
            log.info("sesion abierta: cuenta=%s slot=%s", login, slot.slot_id)
            return session

    def get(self, token: str) -> Session | None:
        session = self._by_token.get(token)
        if session is None:
            return None
        if session.is_expired():
            return None
        session.last_seen = time.time()   # expiracion deslizante
        return session

    def find_by_account(self, login: int, server: str) -> Session | None:
        for session in self._by_token.values():
            if session.login == login and session.server == server and not session.is_expired():
                return session
        return None

    async def drop(self, token: str, reason: str = "logout") -> bool:
        async with self._lock:
            session = self._by_token.pop(token, None)
        if session is None:
            return False
        slot = next((s for s in self._pool.slots if s.slot_id == session.slot_id), None)
        if slot is not None:
            await self._pool.release(slot)
        log.info("sesion cerrada (%s): cuenta=%s slot=%s", reason, session.login, session.slot_id)
        return True

    def all(self) -> list[Session]:
        return [s for s in self._by_token.values() if not s.is_expired()]

    # ------------------------------------------------------------- barrido
    async def start_sweeper(self) -> None:
        self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop_sweeper(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass

    async def _sweep_loop(self) -> None:
        """Libera los slots de las sesiones caducadas.

        Sin esto, un cliente que se desconecta sin hacer logout dejaria su slot
        ocupado para siempre y el pool se agotaria."""
        while True:
            try:
                await asyncio.sleep(config.SWEEP_INTERVAL)
                now = time.time()
                expired = [t for t, s in list(self._by_token.items()) if s.is_expired(now)]
                for token in expired:
                    await self.drop(token, reason="caducada")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- el barrido nunca debe morir
                log.error("fallo en el barrido de sesiones: %s", exc)
