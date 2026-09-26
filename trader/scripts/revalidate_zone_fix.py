"""Steps 2-3 of prompt-fix-invalidacion-obfvg.md: visual re-validation of the
corrected OB/FVG invalidation criterion (touch-then-hold-for-grace-period
instead of any-touch-disqualifies), plus a volume check over each symbol's
FULL H1 history before committing to the 8-year backtest re-run.

Uses the real production `TimeframeAnalysis.from_candles` directly (the
fixed `apply_mitigation` with the grace period) -- not a reimplementation.
Writes logs/zone_fix_revalidation_report.md and (via SendUserFile) focused
example charts for at least 2 symbols.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.engine import TimeframeAnalysis

TIMEFRAME = "H1"
COUNT = 20000  # matches what run_backtest.py fetches for H1 -- the FULL history this phase cares about


def _eligible(zones) -> list:
    return [z for z in zones if z.confirmed_at is not None]


def _respected_then_confirmed(zones) -> list:
    """The genuinely NEW behavior worth eyeballing: zones that were touched
    (not the trivial never-touched case) and still ended up confirmed."""
    return [z for z in zones if z.confirmed_at is not None and z.mitigated_at is not None]


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    per_symbol = {}
    try:
        for symbol in config.symbols:
            df = client.candles(symbol, TIMEFRAME, COUNT, timeout=120)
            analysis = TimeframeAnalysis.from_candles(df, TIMEFRAME, config)
            per_symbol[symbol] = (df, analysis)
            print(f"{symbol} {TIMEFRAME}: {len(df)} velas, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")
    finally:
        client.logout()

    lines = [
        "# Re-validacion del fix de invalidacion OB/FVG -- volumen sobre historico completo",
        "",
        f"grace period configurado: {config.zone_lifecycle.invalidation_grace_m15_candles} velas M15 "
        f"(~{config.zone_lifecycle.invalidation_grace_m15_candles * 15} min)",
        "",
        "| Simbolo | Velas H1 | OB elegibles | OB respetadas-y-confirmadas | FVG elegibles | FVG respetadas-y-confirmadas |",
        "|---|---|---|---|---|---|",
    ]
    for symbol, (df, analysis) in per_symbol.items():
        ob_eligible = _eligible(analysis.order_blocks)
        ob_respected = _respected_then_confirmed(analysis.order_blocks)
        fvg_eligible = _eligible(analysis.fvgs)
        fvg_respected = _respected_then_confirmed(analysis.fvgs)
        lines.append(
            f"| {symbol} | {len(df)} | {len(ob_eligible)} | {len(ob_respected)} | {len(fvg_eligible)} | {len(fvg_respected)} |"
        )

    report = "\n".join(lines) + "\n"
    logs_dir = Path(config.signals_log_path).parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "zone_fix_revalidation_report.md").write_text(report, encoding="utf-8")
    print()
    print(report)

    # Visual re-validation on at least 2 symbols, per the phase's explicit ask.
    chart_symbols = list(config.symbols)[:2]
    for symbol in chart_symbols:
        df, analysis = per_symbol[symbol]
        examples = _respected_then_confirmed(analysis.order_blocks)[:3] + _respected_then_confirmed(analysis.fvgs)[:3]
        out_path = logs_dir / f"zone_fix_examples_{symbol}.html"
        _build_focused_charts(df, examples, out_path)


def _build_focused_charts(df, examples, out_path: Path) -> None:
    ts_to_idx = {t: i for i, t in enumerate(df["timestamp"])}
    pad_bars = 15
    sections = []
    for i, z in enumerate(examples, start=1):
        start_idx = ts_to_idx.get(z.timestamp)
        if start_idx is None:
            continue
        end_ts = z.broken_at if z.broken_at is not None else df["timestamp"].iloc[-1]
        end_idx = ts_to_idx.get(end_ts, len(df) - 1)
        lo, hi = max(0, start_idx - pad_bars), min(len(df), end_idx + pad_bars)
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
            line=dict(width=2, color="black"), fillcolor="rgba(0,150,0,0.25)", layer="below",
        )
        fig.add_vline(x=z.mitigated_at, line_dash="dot", line_color="blue", annotation_text="toque")
        fig.add_vline(x=z.confirmed_at, line_dash="dot", line_color="green", annotation_text="confirmada")
        fig.update_layout(
            title=f"{z.kind} #{i} -- {z.direction}, formada {z.timestamp}, tocada {z.mitigated_at}, "
            f"confirmada {z.confirmed_at}, rota {z.broken_at}",
            xaxis_rangeslider_visible=False, height=420, margin=dict(t=60, b=20),
        )
        sections.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html = (
        "<html><head><title>Re-validacion fix OB/FVG</title>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head><body>'
        "<h2>Zonas tocadas que sobrevivieron el periodo de gracia y quedaron confirmadas</h2>"
        + "".join(f"<div>{s}</div><hr/>" for s in sections)
        + "</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Grafico guardado: {out_path}")


if __name__ == "__main__":
    main()
