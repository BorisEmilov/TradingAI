"""Parte 1 de prompt-auditoria-detectores-y-desglose.md: audits whether the H1
OB/FVG detectors are under-counting zones relative to a looser ICT-literature
reference criterion, isolating which specific threshold (displacement
minimum, invalidation-on-first-touch) excludes the most zones.

Reuses the REAL production detector functions with varied parameters (never
a reimplementation) so "current" vs "loose" is a fair, direct comparison:

  - OB displacement: production requires the breaking candle's range to be
    >= 1.5x ATR(H1) to even form an OB candidate ("loose" = 0.0x, i.e. any
    confirmed BOS/CHoCH creates one, per the plain ICT definition which has
    no displacement-size threshold as part of the base concept).
  - Eligibility as a POI: production excludes a zone from POI search the
    moment it's touched ONCE (`mitigated`), even if price only grazed it and
    the zone was never invalidated ("loose" = only exclude zones that were
    fully INVERTED -- closed all the way through the opposite boundary).
  - FVG gap size: already unrestricted in production (min_gap_pct=0.0), so
    only the eligibility lever applies there.

Does NOT modify any production detector or config. Writes
logs/ob_fvg_audit_report.md and (via SendUserFile) an HTML chart marking
example zones production excludes that the loose criterion would keep.

HISTORICAL: this is the diagnostic that led to prompt-fix-invalidacion-obfvg.md
-- `apply_mitigation` now takes the grace-period fix directly (see
trader/detectors/common.py), so "production" here (grace_period=0) means
"invalidate on ANY close-through, no matter how fast" -- close to but not
exactly the pre-fix behavior (which never even looked past the first touch
for a break). Kept for the historical record, not meant to be re-run as
today's production-vs-loose comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go

from trader.config import load_config
from trader.detectors.common import apply_mitigation
from trader.detectors.fvg import detect_fvg
from trader.detectors.indicators import atr as atr_fn
from trader.detectors.order_blocks import detect_order_blocks
from trader.detectors.structure import detect_structure_breaks, detect_swings
from trader.events import MarketEvent
from trader.gateway_client import PythonGetawayClient

SYMBOL = "EURUSD"
TIMEFRAME = "H1"
COUNT = 17000  # ~2 years of H1, matches the phase's "1-2 years" suggestion


def _inverted_origin_timestamps(inversions: list[MarketEvent]) -> set:
    return {inv.meta["origin_timestamp"] for inv in inversions}


def _eligible_not_mitigated(zones: list[MarketEvent]) -> list[MarketEvent]:
    return [z for z in zones if not z.mitigated]


def _eligible_not_inverted(zones: list[MarketEvent], inverted_origins: set) -> list[MarketEvent]:
    return [z for z in zones if z.timestamp not in inverted_origins]


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        df = client.candles(SYMBOL, TIMEFRAME, COUNT, timeout=120)
    finally:
        client.logout()
    print(f"{SYMBOL} {TIMEFRAME}: {len(df)} velas, {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    swings = detect_swings(df, TIMEFRAME, config.structure.swing_left_bars, config.structure.swing_right_bars)
    structure_events = detect_structure_breaks(df, TIMEFRAME, swings)
    atr_series = atr_fn(df, period=config.structure.atr_period)

    # --- Order Blocks: 2x2 (displacement x eligibility) ---
    raw_ob_current = detect_order_blocks(df, TIMEFRAME, structure_events, atr_series, config.structure.displacement_atr_multiple)
    raw_ob_loose = detect_order_blocks(df, TIMEFRAME, structure_events, atr_series, displacement_atr_multiple=0.0)

    ob_current, ob_current_inv = apply_mitigation(raw_ob_current, df, TIMEFRAME, pd.Timedelta(0))
    ob_loose, ob_loose_inv = apply_mitigation(raw_ob_loose, df, TIMEFRAME, pd.Timedelta(0))

    ob_current_inv_origins = _inverted_origin_timestamps(ob_current_inv)
    ob_loose_inv_origins = _inverted_origin_timestamps(ob_loose_inv)

    ob_v1 = _eligible_not_mitigated(ob_current)  # PRODUCTION
    ob_v2 = _eligible_not_inverted(ob_current, ob_current_inv_origins)  # loose invalidation only
    ob_v3 = _eligible_not_mitigated(ob_loose)  # loose displacement only
    ob_v4 = _eligible_not_inverted(ob_loose, ob_loose_inv_origins)  # loose both

    # --- FVG: gap size already unrestricted, only eligibility lever applies ---
    raw_fvg = detect_fvg(df, TIMEFRAME, config.fvg.min_gap_pct)
    fvg, fvg_inv = apply_mitigation(raw_fvg, df, TIMEFRAME, pd.Timedelta(0))
    fvg_inv_origins = _inverted_origin_timestamps(fvg_inv)

    fvg_v1 = _eligible_not_mitigated(fvg)  # PRODUCTION
    fvg_v2 = _eligible_not_inverted(fvg, fvg_inv_origins)  # loose invalidation

    lines = [
        f"# Auditoria de detectores OB/FVG -- {SYMBOL} {TIMEFRAME}",
        "",
        f"Muestra: {len(df)} velas H1, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}",
        f"Zonas OB totales detectadas (antes de elegibilidad): {len(raw_ob_current)} (displacement actual=1.5x ATR), "
        f"{len(raw_ob_loose)} (displacement=0x, cualquier ruptura confirmada)",
        f"Zonas FVG totales detectadas: {len(raw_fvg)} (ya sin umbral de tamano en produccion)",
        "",
        "## Order Blocks -- matriz 2x2 (desplazamiento x elegibilidad)",
        "",
        "| Variante | Desplazamiento minimo | Elegibilidad | Zonas elegibles |",
        "|---|---|---|---|",
        f"| V1 (produccion) | 1.5x ATR | no tocada nunca | {len(ob_v1)} |",
        f"| V2 (solo afloja invalidacion) | 1.5x ATR | no invertida | {len(ob_v2)} |",
        f"| V3 (solo afloja desplazamiento) | 0x (cualquiera) | no tocada nunca | {len(ob_v3)} |",
        f"| V4 (ambos aflojados) | 0x (cualquiera) | no invertida | {len(ob_v4)} |",
        "",
        f"Zonas adicionales solo por aflojar invalidacion (V2-V1): +{len(ob_v2) - len(ob_v1)}",
        f"Zonas adicionales solo por aflojar desplazamiento (V3-V1): +{len(ob_v3) - len(ob_v1)}",
        f"Zonas adicionales aflojando ambos (V4-V1): +{len(ob_v4) - len(ob_v1)}",
        "",
        "## Fair Value Gaps -- elegibilidad (tamano de gap ya sin restriccion)",
        "",
        "| Variante | Elegibilidad | Zonas elegibles |",
        "|---|---|---|",
        f"| V1 (produccion) | no tocada nunca | {len(fvg_v1)} |",
        f"| V2 (afloja invalidacion) | no invertida | {len(fvg_v2)} |",
        "",
        f"Zonas adicionales solo por aflojar invalidacion (V2-V1): +{len(fvg_v2) - len(fvg_v1)}",
    ]

    dominant = "invalidacion (mitigated-on-first-touch)" if (len(ob_v2) - len(ob_v1)) >= (len(ob_v3) - len(ob_v1)) else "desplazamiento minimo (1.5x ATR)"
    lines += [
        "",
        "## Condicion mas restrictiva identificada",
        "",
        f"Para Order Blocks, la condicion que mas zonas excluye es: **{dominant}**.",
        f"Para FVG, la unica palanca aplicable (tamano de gap ya es 0) es la invalidacion, "
        f"que por si sola libera {len(fvg_v2) - len(fvg_v1)} zonas adicionales.",
    ]

    report = "\n".join(lines) + "\n"
    logs_dir = Path(config.signals_log_path).parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "ob_fvg_audit_report.md").write_text(report, encoding="utf-8")
    print()
    print(report)

    # --- Pick concrete examples: zones production excludes (mitigated) but the
    # loose invalidation criterion keeps (not inverted), same displacement --
    # isolates the invalidation lever specifically, the cleanest "before/after".
    ob_examples = [z for z in ob_v2 if z not in ob_v1][:5]
    fvg_examples = [z for z in fvg_v2 if z not in fvg_v1][:5]

    _build_focused_charts(df, ob_examples, fvg_examples, out_path=logs_dir / "ob_fvg_audit_examples.html")


def _build_focused_charts(df, ob_examples, fvg_examples, out_path: Path) -> None:
    """One small, zoomed-in candlestick per example (not one 17000-bar mega
    chart) -- each case needs to be individually inspectable, per the phase's
    explicit ask for 5-10 concrete visual cases, not a single overwhelming view.
    """
    ts_to_idx = {t: i for i, t in enumerate(df["timestamp"])}
    pad_bars = 15

    sections = []
    for label, examples, fill in (("Order Block", ob_examples, "rgba(255,165,0,0.3)"), ("FVG", fvg_examples, "rgba(128,0,128,0.3)")):
        for i, z in enumerate(examples, start=1):
            start_idx = ts_to_idx.get(z.timestamp)
            if start_idx is None:
                continue
            end_ts = z.mitigated_at if z.mitigated_at is not None else df["timestamp"].iloc[-1]
            end_idx = ts_to_idx.get(end_ts, len(df) - 1)
            lo = max(0, start_idx - pad_bars)
            hi = min(len(df), end_idx + pad_bars)
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
                line=dict(width=2, color="black"), fillcolor=fill, layer="below",
            )
            fig.update_layout(
                title=f"{label} #{i} -- {z.direction}, formada {z.timestamp}, tocada {z.mitigated_at} (no invertida)",
                xaxis_rangeslider_visible=False, height=420, margin=dict(t=40, b=20),
            )
            sections.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html = (
        "<html><head><title>Auditoria OB/FVG -- ejemplos</title>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head><body>'
        "<h2>Zonas que produccion excluye (tocadas una vez) pero el criterio laxo mantiene (no invertidas)</h2>"
        + "".join(f"<div>{s}</div><hr/>" for s in sections)
        + "</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Grafico guardado: {out_path}")


if __name__ == "__main__":
    main()
