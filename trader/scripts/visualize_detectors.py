"""Fase 2 visual validation (prompt-implementacion-agente-trading.md): plot every
detector's output over a real candlestick chart so each one can be checked by
eye before it's trusted in the pipeline.

Reuses `TimeframeAnalysis` -- the exact same code path `MultiTimeframePipeline`
runs in production -- rather than a separate plotting-only reimplementation,
so what you see here is guaranteed to be what the pipeline actually sees.

Usage:
    python scripts/visualize_detectors.py --symbol EURUSD --timeframe H1 --count 300
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go

from trader.config import load_config
from trader.events import MarketEvent
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.engine import TimeframeAnalysis

_ZONE_COLOR = {
    "bullish": "rgba(0,150,0,0.15)",
    "bearish": "rgba(200,0,0,0.15)",
}
_INVERTED_ZONE_COLOR = {
    "bullish": "rgba(0,150,0,0.35)",
    "bearish": "rgba(200,0,0,0.35)",
}
_POINT_STYLE = {
    "bos": dict(symbol="triangle-up", color="blue"),
    "choch": dict(symbol="x", color="purple"),
    "liquidity_sweep_bullish": dict(symbol="star", color="green"),
    "liquidity_sweep_bearish": dict(symbol="star", color="red"),
    "turtle_soup_bullish": dict(symbol="diamond", color="green"),
    "turtle_soup_bearish": dict(symbol="diamond", color="red"),
    "sharp_turn": dict(symbol="hourglass", color="orange"),
    "equal_highs": dict(symbol="line-ew", color="red"),
    "equal_lows": dict(symbol="line-ew", color="green"),
    "elliott_impulse_context": dict(symbol="star-diamond", color="black"),
}


def _zone_shape(z: MarketEvent, x_end) -> dict:
    inverted = z.kind.startswith("inverted_")
    palette = _INVERTED_ZONE_COLOR if inverted else _ZONE_COLOR
    x1 = z.mitigated_at if z.mitigated_at is not None else x_end
    return dict(
        type="rect", xref="x", yref="y",
        x0=z.timestamp, x1=x1, y0=z.price_low, y1=z.price_high,
        line=dict(width=1, color=palette[z.direction]), fillcolor=palette[z.direction], layer="below",
    )


def _point_trace(events: list[MarketEvent], kind: str) -> go.Scatter | None:
    matching = [e for e in events if e.kind == kind]
    if not matching:
        return None
    style = _POINT_STYLE.get(kind, dict(symbol="circle", color="gray"))
    return go.Scatter(
        x=[e.timestamp for e in matching],
        y=[e.price for e in matching],
        mode="markers",
        name=kind,
        marker=dict(symbol=style["symbol"], color=style["color"], size=11, line=dict(width=1, color="black")),
        text=[f"{e.kind} ({e.direction})<br>{e.meta}" for e in matching],
        hoverinfo="text+x+y",
    )


def build_figure(df, analysis: TimeframeAnalysis, symbol: str, timeframe: str) -> go.Figure:
    x_end = df["timestamp"].iloc[-1]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"
            )
        ]
    )

    shapes = [_zone_shape(z, x_end) for z in (analysis.order_blocks + analysis.fvgs)]
    fig.update_layout(shapes=shapes)

    point_kinds = [
        "bos", "choch", "liquidity_sweep_bullish", "liquidity_sweep_bearish",
        "turtle_soup_bullish", "turtle_soup_bearish", "sharp_turn",
        "equal_highs", "equal_lows", "elliott_impulse_context",
    ]
    for kind in point_kinds:
        trace = _point_trace(analysis.all_events(), kind)
        if trace is not None:
            fig.add_trace(trace)

    for sr in analysis.support_resistance:
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=sr.timestamp, x1=x_end, y0=sr.price, y1=sr.price,
            line=dict(width=1, color="gray", dash="dot"),
        )

    fig.update_layout(
        title=f"{symbol} {timeframe} -- deteccion verificada visualmente ({len(df)} velas)",
        xaxis_title="Tiempo (UTC)", yaxis_title="Precio",
        xaxis_rangeslider_visible=False, height=800, legend=dict(orientation="h"),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1", choices=["D1", "H4", "H1", "M15"])
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--out", default=None, help="Ruta del HTML de salida (default: logs/viz_<symbol>_<tf>.html)")
    parser.add_argument("--no-open", action="store_true", help="No abrir el navegador automaticamente")
    args = parser.parse_args()

    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        df = client.candles(args.symbol, args.timeframe, args.count)
    finally:
        client.logout()

    analysis = TimeframeAnalysis.from_candles(df, args.timeframe, config)
    fig = build_figure(df, analysis, args.symbol, args.timeframe)

    out_path = Path(args.out) if args.out else Path("logs") / f"viz_{args.symbol}_{args.timeframe}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True)
    print(f"Grafico guardado en: {out_path.resolve()}")

    if not args.no_open:
        webbrowser.open(f"file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
