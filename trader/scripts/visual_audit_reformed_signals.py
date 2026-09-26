"""Auditoria visual de una muestra de senales del sistema reformulado
(reemplaza el ejercicio manual de graficos en blanco, abandonado per
prompt-implementar-y-validar-edge.md): toma unas pocas senales por
simbolo/razon dominante de logs/reformed_signals.jsonl, dibuja la ventana
M15 real alrededor de cada una con las lineas de entrada/SL/parcial/TP para
verificacion visual rapida -- ventana ACOTADA (no todo el historico) para no
repetir el bug de archivos de 20MB de la auditoria de zonas LTF.

Usage: python scripts/visual_audit_reformed_signals.py [--per-group 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trader.config import load_config
from trader.events import TF_DURATION
from trader.gateway_client import PythonGetawayClient

WINDOW_BEFORE = 40
WINDOW_AFTER = 60
M15_COUNT = 50000


def _load_signals(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def _select_sample(signals: list[dict], per_group: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for s in signals:
        key = (s["symbol"], s["dominant_reason_kind"], s["direction"])
        groups.setdefault(key, []).append(s)
    sample = []
    for key, group in sorted(groups.items()):
        step = max(1, len(group) // per_group)
        sample.extend(group[::step][:per_group])
    return sample


def _chart_for_signal(s: dict, m15_df: pd.DataFrame) -> go.Figure | None:
    generated_at = pd.Timestamp(s["generated_at"])
    close_ts = m15_df["timestamp"] + TF_DURATION["M15"]
    matches = close_ts[close_ts == generated_at]
    if matches.empty:
        return None
    anchor_idx = matches.index[0]

    start = max(0, anchor_idx - WINDOW_BEFORE)
    end = min(len(m15_df), anchor_idx + WINDOW_AFTER)
    window = m15_df.iloc[start:end]

    fig = go.Figure(
        data=[go.Candlestick(
            x=window["timestamp"], open=window["open"], high=window["high"], low=window["low"], close=window["close"],
            name="OHLC",
        )]
    )
    x0, x1 = window["timestamp"].iloc[0], window["timestamp"].iloc[-1]
    lines = [
        ("entry", s["entry_reference_price"], "black"),
        ("SL", s["original_sl"], "red"),
        ("TP final", s["final_target"], "green"),
    ]
    if s["partial_target"] is not None:
        lines.append(("parcial", s["partial_target"], "orange"))
    for label, price, color in lines:
        fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=price, y1=price, line=dict(width=1.5, color=color, dash="dash"))
        fig.add_annotation(x=x1, y=price, text=label, showarrow=False, font=dict(color=color, size=10), xanchor="left")
    fig.add_vline(x=generated_at, line=dict(width=1, color="blue", dash="dot"))

    title = (
        f"{s['symbol']} {s['direction']} | regimen={s['regime']} | "
        f"razon={s['dominant_reason_kind']} (pct={s['dominant_reason_strength_percentile']:.1f}) | "
        f"confirmacion pct={s['confirmation_score_percentile']:.1f} | conviccion=x{s['conviction_multiplier']:.2f} | "
        f"R:R planeado=1:{s['planned_risk_reward']:.2f} | {generated_at}"
    )
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=420, showlegend=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--signals-path", default="logs/reformed_signals.jsonl")
    parser.add_argument("--out", default="logs/visual_audit_reformed.html")
    args = parser.parse_args()

    config = load_config()
    signals = _load_signals(Path(args.signals_path))
    sample = _select_sample(signals, args.per_group)
    print(f"Muestra seleccionada: {len(sample)} senales de {len(signals)} totales")

    by_symbol: dict[str, list[dict]] = {}
    for s in sample:
        by_symbol.setdefault(s["symbol"], []).append(s)

    client = PythonGetawayClient(config.gateway)
    client.login()
    figs = []
    try:
        for symbol, group in by_symbol.items():
            print(f"=== {symbol}: fetching M15 ({len(group)} senales en la muestra) ===")
            m15_df = client.candles(symbol, "M15", M15_COUNT, timeout=120)
            for s in group:
                fig = _chart_for_signal(s, m15_df)
                if fig is None:
                    print(f"  AVISO: no se encontro la vela ancla para {s['symbol']} {s['generated_at']} -- omitida")
                    continue
                figs.append(fig)
    finally:
        client.logout()

    print(f"Graficos generados: {len(figs)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>Auditoria visual -- sistema reformulado</title></head><body>\n")
        fh.write(f"<h2>Auditoria visual de una muestra de senales ({len(figs)} de {len(signals)} totales)</h2>\n")
        for i, fig in enumerate(figs):
            fh.write(f"<div id='chart{i}'></div>\n")
            fh.write(fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))
        fh.write("</body></html>\n")

    print(f"Guardado en: {out_path.resolve()}")


if __name__ == "__main__":
    main()
