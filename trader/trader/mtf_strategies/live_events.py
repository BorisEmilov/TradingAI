"""NUEVO -- Prompt 7, punto 1. Tres canales de notificación, cada uno con un
propósito distinto (decisión de diseño, no redundancia accidental):

1. `write_json_event()` -- JSONL detallado, mismo formato que ya usa el
   piloto RSI (`pilot_rsi_xpt_events.jsonl`): un objeto por línea con TODOS
   los campos, pensado para re-procesar/auditar después, no para leer en
   vivo.
2. `write_human_event()` -- NUEVO en este prompt: un archivo de texto plano
   aparte, una línea legible por evento (timestamp, símbolo, estrategia,
   tipo, detalle), pensado explícitamente para `tail -f` -- Boris pidió
   poder ver los movimientos en tiempo real sin parsear JSON.
3. `notify_desktop()` -- notificación de escritorio vía `notify-send` (Linux,
   ya confirmado disponible en este entorno: GNOME + sesión D-Bus activa,
   `notify-send` probado exitosamente antes de escribir este módulo). Best
   effort: si el comando no existe o falla (headless, sin sesión gráfica,
   otro SO sin equivalente instalado), se ignora en silencio y el archivo de
   eventos de texto queda como canal PRINCIPAL garantizado -- nunca se
   asume que la notificación de escritorio llegó. Deliberadamente NO se
   agregó Telegram/email/etc.: son credenciales y superficie de
   configuración nuevas que nadie pidió, y el archivo de texto + escritorio
   ya cubren "enterarse en el momento" sin ese costo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_NOTIFY_SEND_AVAILABLE = shutil.which("notify-send") is not None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_human_line(ts: str, symbol: str, strategy: str, kind: str, detail: str) -> str:
    """Una línea por evento, ancho fijo de columnas para que `tail -f` quede
    alineado y legible a simple vista."""
    return f"[{ts}] {symbol:<8} {strategy:<14} {kind:<20} {detail}"


def write_json_event(path: Path, kind: str, **fields: Any) -> dict:
    row = {"ts": now_iso(), "kind": kind, **fields}
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def write_human_event(path: Path, symbol: str, strategy: str, kind: str, detail: str) -> str:
    line = format_human_line(now_iso(), symbol, strategy, kind, detail)
    with open(path, "a") as f:
        f.write(line + "\n")
    return line


def notify_desktop(title: str, body: str, urgency: str = "normal") -> bool:
    """Best effort -- devuelve True si se pudo lanzar `notify-send`, False si
    no está disponible o falló (nunca lanza una excepción: una notificación
    de escritorio fallida no puede tumbar el piloto)."""
    if not _NOTIFY_SEND_AVAILABLE:
        return False
    try:
        subprocess.run(
            ["notify-send", "--urgency", urgency, "--app-name", "TradingAI MTF", title, body],
            timeout=5, check=False, capture_output=True,
        )
        return True
    except Exception:  # noqa: BLE001 -- una notificación nunca debe interrumpir el piloto
        return False


def log_event(
    json_path: Path, human_path: Path, symbol: str, strategy: str, kind: str, detail: str,
    desktop_title: str | None = None, **json_fields: Any,
) -> None:
    """Escribe los 3 canales de una sola vez -- lo que llama el script del
    piloto en cada evento real (apertura, cierre, ajuste de SL/TP, error,
    heartbeat). `desktop_title` sólo se pasa para eventos que Boris quiere
    ver como notificación de escritorio (apertura/cierre/ajuste) -- no para
    heartbeats ni chequeos rutinarios sin novedad, para no saturarlo de
    popups."""
    write_json_event(json_path, kind, symbol=symbol, strategy=strategy, detail=detail, **json_fields)
    write_human_event(human_path, symbol, strategy, kind, detail)
    print(format_human_line(now_iso(), symbol, strategy, kind, detail))
    if desktop_title is not None:
        notify_desktop(desktop_title, detail)
