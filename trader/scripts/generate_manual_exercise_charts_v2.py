"""prompt-regenerar-ejercicio-manual.md: la clave del set 1 se filtro (Boris
la vio completa) -- regenera un set NUEVO de 25 casos en blanco con fechas
distintas a las 25 ya reveladas, para que el veredicto manual siga siendo
independiente.

CASOS DE SENAL: de los 6 episodios reales conocidos, 3 (EURUSD 2025-07-01,
EURUSD 2026-01-28, EURUSD 2025-04-21) tienen un UNICO timestamp real
registrado en los logs -- no hay forma de mostrar ese mismo episodio con una
fecha genuina distinta, asi que se OMITEN de este set (no se inventa una
fecha falsa). Los otros 3 episodios sI tienen sub-anclajes reales sin usar
en el set 1 (velas de 15-30 min de diferencia dentro del mismo episodio,
tambien real, tambien verificado contra los .jsonl) -- esos 4 sub-anclajes
son los casos de senal de este set. El resto son puntos aleatorios nuevos
(semilla distinta), excluyendo explicitamente los 25 timestamps del set 1
leido de logs/manual_exercise_key.md.

Mismo formato/regla de causalidad que el generador v1: solo velas, sin
anotaciones, cada grafico termina exactamente en el timestamp del caso.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_manual_exercise_charts import (  # noqa: E402
    D1_WINDOW,
    M15_COUNT,
    M15_WINDOW,
    D1_COUNT,
    SYMBOLS,
    _blank_figure,
    _fetch_all,
    build_case_html,
)

from trader.config import load_config
from trader.events import TF_DURATION
from trader.gateway_client import PythonGetawayClient

SEED_V2 = 20260919_2  # semilla distinta a la v1, documentada, no ajustada despues de ver resultados

# Sub-anclajes reales, verificados contra los .jsonl, NO usados como anchor
# en el set 1 (ese usaba el mas temprano de cada cluster) -- ver docstring.
KNOWN_SIGNAL_CASES_V2 = [
    ("GBPUSD", "2025-05-26T21:30:00+00:00", "OB/FVG-fix, M15+newsfilter, ML (mismo episodio que el caso revelado del 21:15, 15 min despues)", "reversal / turtle_soup_bullish->ifvg, GANADORA (+0.70R)"),
    ("GBPUSD", "2025-05-27T06:00:00+00:00", "OB/FVG-fix, M15+newsfilter, ML (mismo episodio, anclaje final del cluster)", "reversal / inverted_fvg_bearish, GANADORA (+0.52R)"),
    ("GBPUSD", "2026-01-28T05:30:00+00:00", "ML (mismo episodio que el caso revelado del 05:15, 15 min despues)", "reversal / inverted_fvg_bearish, perdedora"),
    ("EURUSD", "2025-01-03T06:15:00+00:00", "Fase 7 ML (mismo episodio que el caso revelado del 06:00, 15 min despues)", "counter_bias_untriggered / choch, perdedora"),
]

N_RANDOM_V2 = 21  # 4 senal + 21 aleatorios = 25


def _load_excluded_timestamps(key_path: Path) -> set[pd.Timestamp]:
    excluded = set()
    if not key_path.exists():
        return excluded
    for line in key_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*\d+\s*\|\s*(\w+)\s*\|\s*([\d\-T:+]+)\s*\|", line)
        if m:
            excluded.add(pd.Timestamp(m.group(2)))
    return excluded


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        data = _fetch_all(client)
    finally:
        client.logout()

    logs_dir = Path(config.signals_log_path).parent
    excluded_ts = _load_excluded_timestamps(logs_dir / "manual_exercise_key.md")
    print(f"Timestamps del set 1 excluidos: {len(excluded_ts)}")

    rng = random.Random(SEED_V2)

    cases: list[dict] = []
    for symbol, ts_str, phase, outcome in KNOWN_SIGNAL_CASES_V2:
        ts = pd.Timestamp(ts_str)
        assert ts not in excluded_ts, f"colision inesperada con set 1: {symbol} {ts}"
        cases.append({"symbol": symbol, "as_of": ts, "kind": "signal", "phase": phase, "outcome": outcome})

    known_ts_by_symbol: dict[str, list[pd.Timestamp]] = {}
    for c in cases:
        known_ts_by_symbol.setdefault(c["symbol"], []).append(c["as_of"])

    per_symbol_random_target = {"EURUSD": 7, "GBPUSD": 6, "USDJPY": 8}
    assert sum(per_symbol_random_target.values()) == N_RANDOM_V2

    for symbol, n in per_symbol_random_target.items():
        m15_df = data[symbol]["M15"]
        m15_close_ts = m15_df["timestamp"] + TF_DURATION["M15"]
        lo_bound = 200
        hi_bound = len(m15_df) - 1
        chosen: list[int] = []
        attempts = 0
        while len(chosen) < n and attempts < 8000:
            attempts += 1
            idx = rng.randint(lo_bound, hi_bound)
            as_of = m15_close_ts.iloc[idx]
            if as_of in excluded_ts:
                continue
            too_close = any(abs((as_of - kts).total_seconds()) < 86400 for kts in known_ts_by_symbol.get(symbol, []))
            if too_close:
                continue
            too_close_excluded = any(abs((as_of - ets).total_seconds()) < 3600 for ets in excluded_ts)
            if too_close_excluded:
                continue
            if any(abs(idx - c) < 500 for c in chosen):
                continue
            chosen.append(idx)
            cases.append({"symbol": symbol, "as_of": as_of, "kind": "no_signal", "phase": "-", "outcome": "-"})

    rng.shuffle(cases)

    key_lines = ["# Clave del ejercicio manual v2 -- NO COMPARTIR hasta despues del veredicto", "", "| Caso | Simbolo | Fecha/hora (UTC) | Tipo | Fase de origen | Resultado real |", "|---|---|---|---|---|---|"]
    html_sections = []

    for i, c in enumerate(cases, start=1):
        label = f"Caso {i}"
        try:
            html_sections.append(build_case_html(c["symbol"], c["as_of"], data[c["symbol"]]["D1"], data[c["symbol"]]["M15"], label))
        except ValueError as e:
            print(f"SKIP {label}: {e}")
            continue
        key_lines.append(f"| {i} | {c['symbol']} | {c['as_of'].isoformat()} | {c['kind']} | {c['phase']} | {c['outcome']} |")

    html = (
        "<html><head><title>Ejercicio manual v2 -- casos en blanco</title>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head><body>'
        "<h2>Ejercicio manual (set 2): 25 casos, solo velas, sin anotaciones</h2>"
        "<p>Cada grafico termina exactamente en el momento del caso -- no hay ninguna vela posterior mostrada. "
        "Para cada caso: contexto D1 arriba, zoom M15 (entrada) abajo.</p>"
        + "".join(f"<hr/><div>{s}</div>" for s in html_sections)
        + "</body></html>"
    )
    out_path = logs_dir / "manual_exercise_blank_cases_v2.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Guardado: {out_path} ({size_mb:.1f} MB)")

    key_path = logs_dir / "manual_exercise_key_v2.md"
    key_path.write_text("\n".join(key_lines) + "\n", encoding="utf-8")
    print(f"Clave guardada (NO se imprime ni se comparte): {key_path}")


if __name__ == "__main__":
    main()
