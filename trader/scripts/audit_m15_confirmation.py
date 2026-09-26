"""Auditoria de M15_confirmation (exigencia de frescura exacta), per
prompt-auditoria-m15-confirmation.md.

Isola la palanca M15_confirmation x H1_poi (igual que la auditoria de OB/FVG
isolo invalidacion x desplazamiento) -- deliberadamente NO aplica
D1_bias/session/risk, asi que los totales absolutos de este script no van a
coincidir con logs/backtest_no_signals.jsonl (que si pasa por todo el
pipeline). Lo que importa aqui es la GANANCIA relativa de aflojar la ventana
de frescura, no el conteo absoluto.

Criterio actual (produccion, `_latest_confirmation` en pipeline/engine.py):
el evento de confirmacion (choch/turtle_soup/sharp_turn/inverted_fvg) tiene
que caer EXACTAMENTE en la vela M15 mas reciente cerrada.

Criterio laxo (solo diagnostico, no se toca produccion): el evento puede
haber ocurrido dentro de las ultimas N velas M15 (ventana configurable).

Reusa directamente las piezas reales de produccion: `TimeframeAnalysis`,
`_CONFIRMATION_KINDS`, la condicion causal de elegibilidad de zonas
(confirmed_at/broken_at) copiada literal de `_evaluate_direction` en
pipeline/engine.py (mismo texto, no una reimplementacion del criterio). Solo
la ventana de frescura es el parametro que varia.

No modifica ningun archivo de produccion. Escribe
logs/m15_confirmation_audit_report.md y graficos de ejemplo via SendUserFile.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go

from trader.config import load_config
from trader.detectors.indicators import atr as atr_fn
from trader.events import TF_DURATION, MarketEvent
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.engine import _CONFIRMATION_KINDS, TimeframeAnalysis

SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY")
H1_COUNT = 20000
M15_COUNT = 50000
M15_STEP = TF_DURATION["M15"]
WINDOWS = (2, 3)  # velas adicionales de gracia mas alla de la exacta (30, 45 min)


def _is_confirmation(e: MarketEvent) -> bool:
    return e.kind in _CONFIRMATION_KINDS or e.kind.startswith("inverted_fair_value_gap")


def _poi_zones_by_direction(h1_analysis: TimeframeAnalysis) -> dict[str, list[MarketEvent]]:
    zones = h1_analysis.order_blocks + h1_analysis.fvgs
    out: dict[str, list[MarketEvent]] = {"bullish": [], "bearish": []}
    for z in zones:
        if z.confirmed_at is None:
            continue
        out[z.direction].append(z)
    return out


def _poi_at(zones_dir: list[MarketEvent], as_of: pd.Timestamp, current_price: float, tol: float) -> MarketEvent | None:
    # Misma condicion causal que `_evaluate_direction` en pipeline/engine.py --
    # copiada literal, no reimplementada con otra logica.
    candidates = [
        z
        for z in zones_dir
        if z.confirmed_at <= as_of
        and (z.broken_at is None or as_of < z.broken_at)
        and z.overlaps(current_price * (1 - tol), current_price * (1 + tol))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda z: z.timestamp)


def main() -> None:
    config = load_config()
    tol = config.poi.tolerance_pct / 100.0
    client = PythonGetawayClient(config.gateway)
    client.login()

    per_symbol = {}
    try:
        for symbol in SYMBOLS:
            h1_df = client.candles(symbol, "H1", H1_COUNT, timeout=120)
            m15_df = client.candles(symbol, "M15", M15_COUNT, timeout=120)
            h1_analysis = TimeframeAnalysis.from_candles(h1_df, "H1", config)
            m15_analysis = TimeframeAnalysis.from_candles(m15_df, "M15", config)
            # primary ATR(14) plus a longer-period fallback: verified against the
            # first run's 2 "n/d" cases (#9/#10) that a manual check showed DID
            # have real, significant drift -- ATR(14) landed on NaN/0 for those
            # specific candles (thin-liquidity/flat stretch), which silently
            # dropped them from both the numerator and denominator instead of
            # reporting a real (large) drift. ATR(50) is far less likely to be
            # exactly flat over the same window, so it's tried first as a floor
            # before ever giving up and marking a case genuinely non-evaluable.
            m15_atr = atr_fn(m15_df, period=14)
            m15_atr_fallback = atr_fn(m15_df, period=50)
            per_symbol[symbol] = (h1_df, m15_df, h1_analysis, m15_analysis, m15_atr, m15_atr_fallback)
            print(f"{symbol}: H1 {len(h1_df)} velas, M15 {len(m15_df)} velas")
    finally:
        client.logout()

    report_lines = [
        "# Auditoria de M15_confirmation -- exigencia de frescura exacta",
        "",
        "Aisla la palanca M15_confirmation x H1_poi (sin D1_bias/session/risk) -- "
        "los totales absolutos NO coinciden con logs/backtest_no_signals.jsonl, "
        "lo que importa es la ganancia relativa de aflojar la ventana.",
        "",
        f"tolerancia POI: {config.poi.tolerance_pct}%",
        "",
    ]

    all_examples: list[dict] = []

    for symbol in SYMBOLS:
        h1_df, m15_df, h1_analysis, m15_analysis, m15_atr, m15_atr_fallback = per_symbol[symbol]
        zones_by_dir = _poi_zones_by_direction(h1_analysis)
        pip_size = 0.01 if "JPY" in symbol else 0.0001

        close_ts = m15_df["timestamp"] + M15_STEP
        price_by_ts = dict(zip(close_ts, m15_df["close"]))
        atr_by_ts = dict(zip(close_ts, m15_atr))
        atr_fallback_by_ts = dict(zip(close_ts, m15_atr_fallback))

        confirmation_events = sorted(
            (e for e in m15_analysis.all_events() if _is_confirmation(e)), key=lambda e: e.timestamp
        )
        exact_match_set = {(e.timestamp, e.direction) for e in confirmation_events}

        print(f"{symbol}: {len(confirmation_events)} eventos de confirmacion en M15")

        # baseline (criterio actual, exacto): cuantos de esos eventos exactos
        # de hecho tenian un POI H1 valido a esa hora/precio.
        baseline_hits = 0
        for e in confirmation_events:
            price = price_by_ts.get(e.timestamp)
            if price is None:
                continue
            if _poi_at(zones_by_dir[e.direction], e.timestamp, price, tol) is not None:
                baseline_hits += 1

        report_lines += [f"## {symbol}", "", f"Eventos de confirmacion M15 totales: {len(confirmation_events)}", ""]
        report_lines += [
            f"Criterio EXACTO (produccion hoy): {baseline_hits} de esos eventos tenian POI H1 valido en ese momento "
            "(candidato real de senal, isolando solo esta condicion).",
            "",
        ]

        symbol_examples: list[dict] = []

        for N in WINDOWS:
            gained: dict[tuple[pd.Timestamp, str], dict] = {}
            for e in confirmation_events:
                for k in range(1, N):
                    candidate_ts = e.timestamp + k * M15_STEP
                    key = (candidate_ts, e.direction)
                    if key in exact_match_set:
                        continue  # esta vela ya tiene su propia confirmacion exacta -- no es ganancia
                    price = price_by_ts.get(candidate_ts)
                    if price is None:
                        continue  # gap de fin de semana/feriado -- esa vela no existe
                    poi = _poi_at(zones_by_dir[e.direction], candidate_ts, price, tol)
                    if poi is None:
                        continue
                    prev = gained.get(key)
                    if prev is None or e.timestamp > prev["event"].timestamp:
                        gained[key] = {"event": e, "candidate_ts": candidate_ts, "k": k, "poi": poi, "price": price}

            event_price = {e.timestamp: price_by_ts.get(e.timestamp) for e in confirmation_events}
            operable = 0
            not_evaluable = 0
            drifts = []
            for (candidate_ts, direction), info in gained.items():
                e = info["event"]
                p0 = event_price.get(e.timestamp)
                p1 = info["price"]
                # drift_pips never depends on ATR -- always computable from raw
                # price alone, so a case is never truly "unmeasurable", only
                # "the ATR-normalized version needs a fallback".
                drift_pips = abs(p1 - p0) / pip_size if p0 is not None else None
                atr_ref = atr_by_ts.get(candidate_ts)
                if not (atr_ref and atr_ref > 0):
                    atr_ref = atr_fallback_by_ts.get(candidate_ts)  # floor: fall back to ATR(50)
                drift_atr = (abs(p1 - p0) / atr_ref) if (p0 is not None and atr_ref and atr_ref > 0) else None
                info["drift_pips"] = drift_pips
                info["drift_atr"] = drift_atr
                if drift_atr is not None:
                    drifts.append(drift_atr)
                    if drift_atr <= 0.5:
                        operable += 1
                else:
                    not_evaluable += 1  # excluded from numerator AND denominator below -- never counted as operable

            n_gained = len(gained)
            n_with_drift = sum(1 for d in drifts if d is not None)
            nd_note = f", {not_evaluable} no evaluables ni por ATR(14) ni por ATR(50) (excluidos, no contados como operables)" if not_evaluable else ""
            report_lines.append(
                f"- Ventana N={N} (hasta {(N - 1) * 15} min de gracia): +{n_gained} candidatos adicionales "
                f"(POI valido + confirmacion dentro de la ventana pero no exacta). "
                f"De esos, {operable}/{n_with_drift} con deriva <=0.5x ATR(M15) desde el evento hasta el punto de entrada "
                f"(umbral orientativo, no una regla -- ver casos visuales){nd_note}."
            )

            for (candidate_ts, direction), info in gained.items():
                info["symbol"] = symbol
                info["direction"] = direction
                info["N"] = N
                symbol_examples.append(info)

        report_lines.append("")
        # Solo se ordena/selecciona dentro de los casos MEDIBLES (drift_atr no
        # es None) -- los no evaluables no deben poder colarse en el extremo
        # "alta deriva" solo porque None ordena al final (ese fue exactamente
        # el bug: 2 de los 10 casos originales terminaron ahi por ausencia de
        # dato, no porque su deriva real fuera alta o baja).
        measurable = sorted((d for d in symbol_examples if d["drift_atr"] is not None), key=lambda d: d["drift_atr"])
        not_evaluable = [d for d in symbol_examples if d["drift_atr"] is None]
        # mezcla: algunos con poca deriva (aparentan operables) y algunos con mucha (aparentan ya corridos)
        picks = measurable[:2] + measurable[len(measurable) // 2 : len(measurable) // 2 + 1] + measurable[-2:]
        if not_evaluable:
            picks = picks[:4] + not_evaluable[:1]  # como mucho 1 caso "no evaluable", etiquetado como tal
        all_examples += [p for p in picks if p not in all_examples][:5]

    report = "\n".join(report_lines) + "\n"
    logs_dir = Path(config.signals_log_path).parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "m15_confirmation_audit_report.md").write_text(report, encoding="utf-8")
    print()
    print(report)

    _build_focused_charts(per_symbol, all_examples[:10], logs_dir / "m15_confirmation_audit_examples.html")


def _build_focused_charts(per_symbol: dict, examples: list[dict], out_path: Path) -> None:
    sections = []
    for i, info in enumerate(examples, start=1):
        symbol = info["symbol"]
        _, m15_df, _, _, _, _ = per_symbol[symbol]
        e = info["event"]
        candidate_ts = info["candidate_ts"]
        poi = info["poi"]

        ts_to_idx = {t: idx for idx, t in enumerate((m15_df["timestamp"] + M15_STEP))}
        start_idx = ts_to_idx.get(e.timestamp)
        end_idx = ts_to_idx.get(candidate_ts)
        if start_idx is None or end_idx is None:
            continue
        pad = 10
        lo, hi = max(0, start_idx - pad), min(len(m15_df), end_idx + pad + 1)
        window = m15_df.iloc[lo:hi]

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=window["timestamp"], open=window["open"], high=window["high"], low=window["low"],
                    close=window["close"], name="OHLC",
                )
            ]
        )
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=window["timestamp"].iloc[0], x1=window["timestamp"].iloc[-1],
            y0=poi.price_low, y1=poi.price_high, line=dict(width=1, color="black"),
            fillcolor="rgba(0,150,0,0.15)", layer="below",
        )
        fig.add_vline(x=e.timestamp - TF_DURATION["M15"], line_dash="dot", line_color="blue", annotation_text="confirmacion")
        fig.add_vline(x=candidate_ts - TF_DURATION["M15"], line_dash="dot", line_color="red", annotation_text="entrada (laxo)")
        drift_atr = info.get("drift_atr")
        drift_pips = info.get("drift_pips")
        pips_str = f"{drift_pips:.1f} pips" if drift_pips is not None else "n/d"
        atr_str = f"{drift_atr:.2f}x ATR(M15)" if drift_atr is not None else "no evaluable (ATR14/ATR50 planos)"
        fig.update_layout(
            title=(
                f"#{i} {symbol} -- {e.kind} {info['direction']}, confirmado {e.timestamp}, "
                f"entrada laxa {candidate_ts} (N={info['N']}, k={info['k']} velas de gracia), "
                f"deriva={pips_str} ({atr_str})"
            ),
            xaxis_rangeslider_visible=False, height=420, margin=dict(t=60, b=20),
        )
        sections.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html = (
        "<html><head><title>Auditoria M15_confirmation -- ejemplos</title>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head><body>'
        "<h2>Candidatos que el criterio laxo rescata pero el exacto rechaza</h2>"
        "<p>Linea azul = vela de confirmacion original. Linea roja = vela donde se generaria la entrada bajo el criterio laxo.</p>"
        + "".join(f"<div>{s}</div><hr/>" for s in sections)
        + "</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Grafico guardado: {out_path}")


if __name__ == "__main__":
    main()
