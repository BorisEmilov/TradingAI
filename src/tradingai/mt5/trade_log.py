"""Tabla de operaciones del piloto en vivo: log de eventos APERTURA/CIERRE en CSV.

Los 5 procesos de `run_live.py` (uno por simbolo) escriben en el MISMO archivo en
paralelo. Se usa un log de eventos append-only (nunca se edita una fila existente)
en vez de una tabla con "una fila por operacion que se actualiza al cerrar": eso
evitaria una carrera de lectura-modificacion-escritura entre procesos. Cada apertura
y cada cierre son una fila nueva, enlazadas por `ticket` -- se puede reconstruir el
historial completo de cada operacion agrupando por ticket.

`fcntl.flock` evita que dos procesos entrelacen filas al escribir a la vez.
"""

from __future__ import annotations

import csv
import fcntl
from pathlib import Path

COLUMNS = [
    "timestamp", "event", "ticket", "symbol", "direction",
    "price", "sl", "tp", "confidence", "lot_size", "profit",
]


def append_trade_event(path: str | Path, **fields) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {col: fields.get(col, "") for col in COLUMNS}

    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            write_header = path.stat().st_size == 0
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
