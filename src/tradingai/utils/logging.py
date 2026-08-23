"""Configuracion centralizada de logging (loguru) para todo el proyecto."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(log_level: str = "INFO", logs_dir: str | Path = "logs") -> "logger":
    """Configura loguru con salida a consola y a fichero rotativo. Idempotente."""
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True)
    logger.add(
        logs_path / "tradingai_{time:YYYY-MM-DD}.log",
        level=log_level,
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )

    _CONFIGURED = True
    return logger
