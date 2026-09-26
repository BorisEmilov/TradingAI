"""Backtest de la arquitectura reformulada de 4 capas
(logs/reformulacion_diseno_capas.md) sobre historia real, mismos
simbolos/periodo/cuenta que cada baseline anterior de este proyecto.

Escribe:
  - logs/reformed_signals.jsonl  -- una linea por senal generada (ReformedSignal)
  - logs/reformed_trades.jsonl   -- una linea por operacion simulada (ManagedTradeResult)
  - logs/reformed_backtest_report.md -- metricas agregadas + tabla comparativa vs. TODOS los baselines anteriores

Usage: python scripts/run_backtest_reformed.py [--count 50000]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.metrics import breakdown_by, compute_metrics
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.management import BASE_RISK_PCT, simulate_managed_trade
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.reformed import generate_reformed_signals

TIMEFRAMES = ("D1", "H1", "M15")
DEFAULT_COUNTS = {"D1": 1500, "H1": 20000, "M15": 50000}

# Baselines de fases anteriores de este proyecto, mismos simbolos/periodo/cuenta
# (ver logs/ de cada fase para el detalle completo) -- para la tabla comparativa
# final, no para ningun calculo.
PRIOR_BASELINES = [
    ("H4 original (antes de esta linea de trabajo)", 3),
    ("Sin H4, sin fix de piso de SL", 8),
    ("Sin H4, con fix de piso de SL", 1),
    ("Fix de invalidacion OB/FVG (periodo de gracia)", 3),
    ("M15 window=2 + filtro de noticias", 4),
    ("Sistema de puntuacion ponderada (scoring)", 2),
    ("Fase 7 -- dataset ML de candidatos elegibles (sin modelo entrenado)", 9),
    ("Experimento D1->LTF unico, variante M15", 1),
    ("Experimento D1->LTF unico, variante M30", 0),
]


def _json_default(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=_json_default) + "\n")


def _fmt_metrics(m) -> str:
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return f"n={m.n_trades}, win_rate={m.win_rate:.1%}, expectancy_r={m.expectancy_r:+.3f}, profit_factor={pf}, max_dd_r={m.max_drawdown_r:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None, help="override: same count for every timeframe")
    args = parser.parse_args()

    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    all_signals: list = []
    all_trades: list = []
    volume_by_symbol_direction_kind: dict[tuple[str, str, str], int] = {}

    try:
        for symbol in config.symbols:
            print(f"=== {symbol}: fetching history ===")
            candles = {
                tf: client.candles(symbol, tf, args.count or DEFAULT_COUNTS[tf], timeout=120) for tf in TIMEFRAMES
            }
            for tf, df in candles.items():
                print(f"  {tf}: {len(df)} velas, {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")

            info = client.symbol_info(symbol, timeout=30)
            point = float(info["point"])
            cost = estimate_symbol_cost(symbol, point, candles["M15"])

            print(f"=== {symbol}: analizando detectores (D1/H1/M15) ===")
            d1_analysis = TimeframeAnalysis.from_candles(candles["D1"], "D1", config)
            h1_analysis = TimeframeAnalysis.from_candles(candles["H1"], "H1", config)
            m15_analysis = TimeframeAnalysis.from_candles(candles["M15"], "M15", config)

            print(f"=== {symbol}: generando senales (4 capas) ===")
            generated = generate_reformed_signals(
                symbol, d1_analysis, h1_analysis, m15_analysis, config, avg_spread_price=cost.avg_spread_price
            )
            print(f"  {len(generated)} senales generadas")

            for g in generated:
                s = g.signal
                all_signals.append(asdict(s))
                key = (symbol, s.direction, s.dominant_reason_kind)
                volume_by_symbol_direction_kind[key] = volume_by_symbol_direction_kind.get(key, 0) + 1

                result = simulate_managed_trade(
                    symbol=symbol,
                    direction=s.direction,
                    regime=s.regime,
                    dominant_reason_kind=s.dominant_reason_kind,
                    signal_bar_idx=g.anchor_bar_idx,
                    m15_df=candles["M15"],
                    m15_analysis=m15_analysis,
                    original_sl=s.original_sl,
                    partial_target=s.partial_target,
                    final_target=s.final_target,
                    symbol_cost=cost,
                    base_risk_pct=BASE_RISK_PCT,
                    conviction_mult=s.conviction_multiplier,
                )
                if result is not None:
                    all_trades.append(result)

            print(f"  {len(all_trades)} operaciones simuladas acumuladas hasta ahora")
    finally:
        client.logout()

    logs_dir = Path(config.signals_log_path).parent
    _write_jsonl(all_signals, logs_dir / "reformed_signals.jsonl")
    _write_jsonl([asdict(t) for t in all_trades], logs_dir / "reformed_trades.jsonl")

    overall = compute_metrics(all_trades)
    by_symbol = breakdown_by(all_trades, lambda t: t.symbol)
    by_direction = breakdown_by(all_trades, lambda t: t.direction)
    by_regime = breakdown_by(all_trades, lambda t: t.regime)
    by_reason = breakdown_by(all_trades, lambda t: t.dominant_reason_kind)
    by_exit_reason = breakdown_by(all_trades, lambda t: t.exit_reason)

    lines = [
        "# Reporte de backtest -- sistema reformulado (4 capas)",
        "",
        f"Simbolos: {', '.join(config.symbols)}",
        f"Total senales generadas (las 4 capas + gates de elegibilidad): {len(all_signals)}",
        f"Total operaciones simuladas: {len(all_trades)}",
        "",
        "## Chequeo de volumen por simbolo/direccion/razon dominante",
        "",
    ]
    for (symbol, direction, kind), count in sorted(volume_by_symbol_direction_kind.items()):
        lines.append(f"- {symbol} / {direction} / {kind}: {count}")

    lines += ["", "## Agregado", "", _fmt_metrics(overall), ""]
    lines += ["## Por simbolo", ""]
    for symbol, m in sorted(by_symbol.items()):
        lines.append(f"- {symbol}: {_fmt_metrics(m)}")
    lines += ["", "## Por direccion", ""]
    for direction, m in sorted(by_direction.items()):
        lines.append(f"- {direction}: {_fmt_metrics(m)}")
    lines += ["", "## Por regimen D1", ""]
    for regime, m in sorted(by_regime.items()):
        lines.append(f"- {regime}: {_fmt_metrics(m)}")
    lines += ["", "## Por razon dominante", ""]
    for reason, m in sorted(by_reason.items()):
        lines.append(f"- {reason}: {_fmt_metrics(m)}")
    lines += ["", "## Por razon de salida", ""]
    for reason, m in sorted(by_exit_reason.items()):
        lines.append(f"- {reason}: {_fmt_metrics(m)}")

    lines += ["", "## Tabla comparativa vs. TODOS los baselines anteriores de este proyecto", ""]
    lines.append("| Configuracion | n operaciones |")
    lines.append("|---|---|")
    for label, n in PRIOR_BASELINES:
        lines.append(f"| {label} | {n} |")
    lines.append(f"| **Sistema reformulado (4 capas) -- este backtest** | **{overall.n_trades}** |")

    report = "\n".join(lines) + "\n"
    (logs_dir / "reformed_backtest_report.md").write_text(report, encoding="utf-8")

    print()
    print(report)


if __name__ == "__main__":
    main()
