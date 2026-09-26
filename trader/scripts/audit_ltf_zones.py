"""Paso 4 de prompt-experimento-d1-30m-15m.md: auditoria visual de las zonas
OB/FVG detectadas en las variantes D1->M15-unico y D1->M30-unico, antes de
confiar en el volumen del chequeo rapido. Mismo rigor que
scripts/audit_ob_fvg_detectors.py y scripts/revalidate_zone_fix.py: casos
concretos, datos crudos, verificando que las zonas sean razonables y no
ruido de timeframe bajo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.experiment_ltf import LTF_15M, LTF_30M, build_ltf_analysis

SYMBOL = "EURUSD"
M15_COUNT = 50000
M30_COUNT = 25000


def _respected_then_confirmed(zones) -> list:
    return [z for z in zones if z.confirmed_at is not None and z.mitigated_at is not None]


def _never_touched(zones) -> list:
    return [z for z in zones if z.confirmed_at is not None and z.mitigated_at is None]


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        m15_df = client.candles(SYMBOL, "M15", M15_COUNT, timeout=120)
        m30_df = client.candles(SYMBOL, "M30", M30_COUNT, timeout=120)
    finally:
        client.logout()

    logs_dir = Path(config.signals_log_path).parent
    for thresholds, df in [(LTF_15M, m15_df), (LTF_30M, m30_df)]:
        analysis = build_ltf_analysis(df, thresholds, config)
        ob_respected = _respected_then_confirmed(analysis.order_blocks)
        ob_fresh = _never_touched(analysis.order_blocks)
        fvg_respected = _respected_then_confirmed(analysis.fvgs)
        fvg_fresh = _never_touched(analysis.fvgs)

        print(f"{thresholds.timeframe}: OB respetadas-y-confirmadas={len(ob_respected)}, OB frescas={len(ob_fresh)}, "
              f"FVG respetadas-y-confirmadas={len(fvg_respected)}, FVG frescas={len(fvg_fresh)}")

        # 2 por categoria (8 total) resulto en un HTML de ~20MB combinado, muy
        # lento/fallido para subir -- un archivo POR VARIANTE, con menos
        # ejemplos (2 por categoria sigue siendo 8... se reduce a 1 por
        # categoria = 4 por variante) para que cada archivo sea liviano.
        examples = ob_respected[:1] + ob_fresh[:1] + fvg_respected[:1] + fvg_fresh[:1]
        sections = _build_charts(df, thresholds.timeframe, examples)

        html = (
            f"<html><head><title>Auditoria zonas D1-{thresholds.timeframe}</title>"
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head><body>'
            f"<h2>Zonas OB/FVG detectadas en la variante D1-{thresholds.timeframe} "
            "(1 ejemplo por categoria: OB respetada-y-confirmada, OB fresca, FVG respetada-y-confirmada, FVG fresca)</h2>"
            + "".join(f"<div>{s}</div><hr/>" for s in sections)
            + "</body></html>"
        )
        out_path = logs_dir / f"ltf_zones_audit_examples_{thresholds.timeframe.lower()}.html"
        out_path.write_text(html, encoding="utf-8")
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Guardado: {out_path} ({size_mb:.1f} MB)")


def _build_charts(df, timeframe: str, examples: list) -> list[str]:
    from trader.events import TF_DURATION

    ts_to_idx = {t: i for i, t in enumerate(df["timestamp"] + TF_DURATION[timeframe])}
    pad = 15
    sections = []
    for i, z in enumerate(examples, start=1):
        start_idx = ts_to_idx.get(z.timestamp)
        if start_idx is None:
            continue
        # Cap the window even for a zone that never broke (or a touch/confirm
        # far in the future) -- an unbounded end_ts (literally "the last
        # candle of the whole 50000-bar dataset") produced a single
        # multi-thousand-candle chart that bloated the file to ~20MB and
        # timed out uploading. 150 bars past formation is plenty to see
        # formation -> touch -> confirmation with context, without needing
        # to render years of irrelevant flat candles after it.
        end_ts = z.broken_at if z.broken_at is not None else df["timestamp"].iloc[-1]
        end_idx = min(ts_to_idx.get(end_ts, len(df) - 1), start_idx + 150)
        lo, hi = max(0, start_idx - pad), min(len(df), end_idx + pad)
        window = df.iloc[lo:hi]

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=window["timestamp"], open=window["open"], high=window["high"], low=window["low"],
                    close=window["close"], name="OHLC",
                )
            ]
        )
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=z.timestamp, x1=end_ts, y0=z.price_low, y1=z.price_high,
            line=dict(width=2, color="black"), fillcolor="rgba(0,150,0,0.2)", layer="below",
        )
        if z.mitigated_at is not None:
            fig.add_vline(x=z.mitigated_at, line_dash="dot", line_color="blue", annotation_text="toque")
        fig.add_vline(x=z.confirmed_at, line_dash="dot", line_color="green", annotation_text="confirmada")
        fig.update_layout(
            title=f"{timeframe} #{i} {z.kind} -- {z.direction}, formada {z.timestamp}, tocada {z.mitigated_at}, "
            f"confirmada {z.confirmed_at}, rota {z.broken_at}",
            xaxis_rangeslider_visible=False, height=420, margin=dict(t=60, b=20),
        )
        sections.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return sections


if __name__ == "__main__":
    main()
