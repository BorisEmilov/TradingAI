"""prompt-graficos-en-blanco-ejercicio-manual.md: genera 25 casos EN BLANCO
(D1 de contexto + M15 de entrada, SOLO velas, sin ninguna zona/linea/anotacion
de ningun detector) para el ejercicio manual del checkpoint 3 de la
reformulacion.

REGLA CRITICA DE CAUSALIDAD DEL EJERCICIO: cada grafico termina EXACTAMENTE
en el timestamp del caso -- no se muestra ni una sola vela posterior. Es la
misma disciplina de causalidad de todo el proyecto, aplicada a un ejercicio
humano en vez de a codigo: se juzga "entrarias aqui" parado en ese momento,
sin ver que pasa despues (ver sesgo de retrospectiva).

6 casos son fechas donde el sistema anterior (en alguna de sus 8+
configuraciones) SI genero una senal -- 19 son puntos aleatorios donde NO
genero nada. Se mezclan y numeran sin orden (Caso 1..25) para que el orden
mismo no delate cual es cual. La clave (que es que) se guarda aparte en
logs/manual_exercise_key.md y NO se imprime ni se resume en la salida de
este script -- se comparte explicitamente solo cuando el usuario lo pida,
despues del veredicto manual.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go

from trader.config import load_config
from trader.events import TF_DURATION
from trader.gateway_client import PythonGetawayClient

D1_COUNT = 1500
M15_COUNT = 50000
D1_WINDOW = 90   # ~3 meses de contexto de regimen
M15_WINDOW = 120  # ~1.25 dias de trading de contexto inmediato
SEED = 20260919  # fijo, documentado, no ajustado despues de ver el resultado

# (symbol, timestamp UTC, fase de origen, resultado real) -- la unica fuente
# de verdad para estos 6 son los logs ya guardados de fases anteriores,
# verificados leyendo los .jsonl directamente, no de memoria.
KNOWN_SIGNAL_CASES = [
    ("EURUSD", "2025-07-01T06:30:00+00:00", "OB/FVG-fix, M15+newsfilter, scoring, ML, LTF-M15 (multiples fases)", "trend / turtle_soup_bullish, perdedora"),
    ("EURUSD", "2026-01-28T06:30:00+00:00", "M15+newsfilter, scoring, ML (multiples fases)", "trend / choch, perdedora"),
    ("GBPUSD", "2025-05-26T21:15:00+00:00", "OB/FVG-fix, M15+newsfilter, ML (multiples fases)", "reversal / turtle_soup_bullish->ifvg, GANADORA (+0.52R a +0.70R segun fase)"),
    ("GBPUSD", "2026-01-28T05:15:00+00:00", "OB/FVG-fix, M15+newsfilter, ML (multiples fases)", "reversal / inverted_fvg_bearish, perdedora"),
    ("EURUSD", "2025-01-03T06:00:00+00:00", "Fase 7 ML (dataset ampliado, categoria nueva)", "counter_bias_untriggered / choch, perdedora"),
    ("EURUSD", "2025-04-21T12:15:00+00:00", "Experimento D1->M15 unico", "trend / inverted_fvg_bearish, perdedora"),
]

N_RANDOM = 19
SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY")


def _fetch_all(client: PythonGetawayClient) -> dict[str, dict[str, pd.DataFrame]]:
    data = {}
    for symbol in SYMBOLS:
        d1_df = client.candles(symbol, "D1", D1_COUNT, timeout=120)
        m15_df = client.candles(symbol, "M15", M15_COUNT, timeout=120)
        data[symbol] = {"D1": d1_df, "M15": m15_df}
        print(f"{symbol}: D1={len(d1_df)} M15={len(m15_df)}")
    return data


def _blank_figure(df: pd.DataFrame, end_idx: int, window: int, title: str) -> go.Figure:
    lo = max(0, end_idx - window + 1)
    view = df.iloc[lo : end_idx + 1]
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=view["timestamp"], open=view["open"], high=view["high"], low=view["low"], close=view["close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=380, margin=dict(t=40, b=20))
    return fig


def build_case_html(symbol: str, as_of: pd.Timestamp, d1_df: pd.DataFrame, m15_df: pd.DataFrame, case_label: str) -> str:
    d1_close_ts = d1_df["timestamp"] + TF_DURATION["D1"]
    m15_close_ts = m15_df["timestamp"] + TF_DURATION["M15"]

    d1_idx = d1_close_ts[d1_close_ts <= as_of].index
    m15_idx = m15_close_ts[m15_close_ts <= as_of].index
    if len(d1_idx) == 0 or len(m15_idx) == 0:
        raise ValueError(f"sin historia suficiente para {symbol} {as_of}")
    d1_end = int(d1_idx[-1])
    m15_end = int(m15_idx[-1])

    fig_d1 = _blank_figure(d1_df, d1_end, D1_WINDOW, f"{case_label} -- contexto D1 (termina en la vela que cierra en/antes de {as_of})")
    fig_m15 = _blank_figure(m15_df, m15_end, M15_WINDOW, f"{case_label} -- entrada M15")

    return fig_d1.to_html(full_html=False, include_plotlyjs=False) + fig_m15.to_html(full_html=False, include_plotlyjs=False)


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        data = _fetch_all(client)
    finally:
        client.logout()

    rng = random.Random(SEED)

    cases: list[dict] = []
    for symbol, ts_str, phase, outcome in KNOWN_SIGNAL_CASES:
        cases.append({"symbol": symbol, "as_of": pd.Timestamp(ts_str), "kind": "signal", "phase": phase, "outcome": outcome})

    known_ts_by_symbol: dict[str, list[pd.Timestamp]] = {}
    for c in cases:
        known_ts_by_symbol.setdefault(c["symbol"], []).append(c["as_of"])

    per_symbol_random_target = {
        "EURUSD": 6, "GBPUSD": 6, "USDJPY": 7,  # USDJPY nunca disparo una senal en ninguna fase -- se compensa con mas casos "sin senal" para que quede representado
    }
    assert sum(per_symbol_random_target.values()) == N_RANDOM

    for symbol, n in per_symbol_random_target.items():
        m15_df = data[symbol]["M15"]
        m15_close_ts = m15_df["timestamp"] + TF_DURATION["M15"]
        lo_bound = 200  # suficiente historia previa para D1_WINDOW/M15_WINDOW
        hi_bound = len(m15_df) - 1
        chosen: list[int] = []
        attempts = 0
        while len(chosen) < n and attempts < 5000:
            attempts += 1
            idx = rng.randint(lo_bound, hi_bound)
            as_of = m15_close_ts.iloc[idx]
            too_close = any(abs((as_of - kts).total_seconds()) < 86400 for kts in known_ts_by_symbol.get(symbol, []))
            if too_close:
                continue
            if any(abs(idx - c) < 500 for c in chosen):  # evita que dos casos aleatorios queden pegados
                continue
            chosen.append(idx)
            cases.append({"symbol": symbol, "as_of": as_of, "kind": "no_signal", "phase": "-", "outcome": "-"})

    rng.shuffle(cases)

    logs_dir = Path(config.signals_log_path).parent
    key_lines = ["# Clave del ejercicio manual -- NO COMPARTIR hasta despues del veredicto", "", "| Caso | Simbolo | Fecha/hora (UTC) | Tipo | Fase de origen | Resultado real |", "|---|---|---|---|---|---|"]
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
        "<html><head><title>Ejercicio manual -- casos en blanco</title>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head><body>'
        "<h2>Ejercicio manual: 25 casos, solo velas, sin anotaciones</h2>"
        "<p>Cada grafico termina exactamente en el momento del caso -- no hay ninguna vela posterior mostrada. "
        "Para cada caso: contexto D1 arriba, zoom M15 (entrada) abajo.</p>"
        + "".join(f"<hr/><div>{s}</div>" for s in html_sections)
        + "</body></html>"
    )
    out_path = logs_dir / "manual_exercise_blank_cases.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Guardado: {out_path} ({size_mb:.1f} MB)")

    key_path = logs_dir / "manual_exercise_key.md"
    key_path.write_text("\n".join(key_lines) + "\n", encoding="utf-8")
    print(f"Clave guardada (NO se imprime ni se comparte): {key_path}")


if __name__ == "__main__":
    main()
