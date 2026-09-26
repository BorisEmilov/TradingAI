"""Prompt 7, punto 3: comando de PARADA para `run_mtf_pilot.py`.

Escribe un archivo "stop flag" que el bucle chequea al PRINCIPIO de cada
ciclo (cada POLL_SECONDS, 5 min) -- termina el ciclo en curso de forma
ordenada y sale, nunca corta a mitad de una evaluación. Elegido en vez de
SIGTERM por ser más simple de verificar desde afuera (`ls logs/`) y no
depender de cómo esté corriendo el proceso (foreground, nohup, systemd...);
misma filosofía de persistencia en disco que ya usa el resto del proyecto
(JSON state, JSONL events) en vez de mecanismos de proceso.

IMPORTANTE -- esto SOLO detiene la evaluación de señales NUEVAS. Las
posiciones ya abiertas no se tocan: siguen protegidas por sus órdenes
reales de SL/TP en MT5 (ver `mtf_strategies/live_state.py`) y se resuelven
solas (el bróker las cierra en SL o TP2, o el bot las retoma si se
reinicia) aunque este proceso esté detenido. "Parar el análisis" no es
"cerrar todo".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STOP_FLAG_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_pilot_stop.flag"
STATE_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_pilot_state.json"


def main() -> None:
    STOP_FLAG_PATH.parent.mkdir(exist_ok=True)
    STOP_FLAG_PATH.write_text("stop requested\n")
    print(f"Señal de parada escrita en {STOP_FLAG_PATH}.")
    print("El piloto terminará su ciclo en curso (hasta ~5 min) y saldrá solo -- revisa "
          "logs/mtf_pilot_events.txt para confirmar 'pilot_stopping' / 'session_closed'.")

    if STATE_PATH.exists():
        import json
        with open(STATE_PATH) as f:
            state = json.load(f)
        n_open = len(state.get("open_positions", {}))
        if n_open:
            print(f"\nAVISO: hay {n_open} posición(es) abierta(s) trackeada(s). NO se cierran -- "
                  "siguen protegidas por SL/TP reales en MT5 y se resuelven solas (o el bot las "
                  "retoma si se reinicia). Para cerrarlas manualmente, usar el terminal MT5 o el "
                  "endpoint /trading/close-all del gateway directamente -- este comando no lo hace.")
        else:
            print("\nNo hay posiciones abiertas trackeadas en este momento.")


if __name__ == "__main__":
    main()
