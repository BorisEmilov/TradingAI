"""Vigilancia termica para jobs de CPU largos (entrenamiento/walk-forward).

Descubierto en vivo el 2026-08-24/25: esta maquina puede subir de <70C a 100C en
menos de 2 minutos bajo carga sostenida de un solo proceso multi-hilo, y limitar los
hilos via variables de entorno (OMP_NUM_THREADS etc.) no siempre alcanza a evitarlo
por si solo. Este modulo le da al propio script la capacidad de vigilar la
temperatura y pausar solo si hace falta, en vez de depender de una alerta externa
que llega demasiado tarde para reaccionar a tiempo.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

_THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")


def read_cpu_temp_c() -> float | None:
    """Temperatura de CPU en C, o None si no se puede leer (ej. no-Linux, sin permiso)."""
    try:
        return int(_THERMAL_ZONE.read_text().strip()) / 1000
    except (OSError, ValueError):
        return None


def wait_for_safe_temp(max_temp_c: float = 80.0, poll_seconds: float = 5.0, max_wait_seconds: float = 300.0) -> None:
    """Bloquea hasta que la temperatura este por debajo de `max_temp_c`.

    Si no se puede leer la temperatura (ej. `/sys/class/thermal` no disponible),
    no bloquea -- mejor continuar sin esta proteccion que colgarse para siempre.
    Si sigue caliente tras `max_wait_seconds`, continua igual (evita un cuelgue
    indefinido si el sensor esta mal o el enfriado es genuinamente lento) pero deja
    un aviso claro en el log.
    """
    temp = read_cpu_temp_c()
    if temp is None:
        return

    waited = 0.0
    while temp is not None and temp >= max_temp_c and waited < max_wait_seconds:
        logger.warning(f"Temperatura CPU {temp:.0f}C >= {max_temp_c:.0f}C, pausando {poll_seconds:.0f}s para enfriar...")
        time.sleep(poll_seconds)
        waited += poll_seconds
        temp = read_cpu_temp_c()

    if temp is not None and temp >= max_temp_c:
        logger.warning(f"Temperatura sigue en {temp:.0f}C tras esperar {waited:.0f}s, se continua de todas formas.")
