"""prompt-3-chequeos-antes-de-cerrar.md, chequeo 3: expectancy con y sin la
regla de parcial en el primer nivel intermedio.

La regla de "cerrar parcial en el primer nivel intermedio" (Capa 4) se penso
para la densidad de niveles H1 de la arquitectura anterior. El 82.6% de
operaciones terminando en trailing_stop_post_partial (backtest completo,
logs/reformed_backtest_report.md) sugiere que con niveles M15 -- mucho mas
densos -- la regla se dispara casi de inmediato, dejando corta la mayoria de
las operaciones. Corre el mismo backtest completo en dos variantes, MISMA
seleccion de entrada (mismas senales), solo cambiando si se aplica el
parcial: con la regla (baseline) vs. sin ella (toda la posicion corre a
SL/TP final sin cierre intermedio).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.metrics import compute_metrics
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.management import BASE_RISK_PCT, simulate_managed_trade
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.reformed import generate_reformed_signals

TIMEFRAMES = ("D1", "H1", "M15")
DEFAULT_COUNTS = {"D1": 1500, "H1": 20000, "M15": 50000}


def _fmt_metrics(m) -> str:
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return f"n={m.n_trades}, win_rate={m.win_rate:.1%}, expectancy_r={m.expectancy_r:+.3f}, profit_factor={pf}, max_dd_r={m.max_drawdown_r:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    with_partial_trades = []
    without_partial_trades = []

    try:
        for symbol in config.symbols:
            print(f"=== {symbol}: fetching + generando senales ===")
            candles = {tf: client.candles(symbol, tf, args.count or DEFAULT_COUNTS[tf], timeout=120) for tf in TIMEFRAMES}
            info = client.symbol_info(symbol, timeout=30)
            point = float(info["point"])
            cost = estimate_symbol_cost(symbol, point, candles["M15"])

            d1_analysis = TimeframeAnalysis.from_candles(candles["D1"], "D1", config)
            h1_analysis = TimeframeAnalysis.from_candles(candles["H1"], "H1", config)
            m15_analysis = TimeframeAnalysis.from_candles(candles["M15"], "M15", config)

            generated = generate_reformed_signals(
                symbol, d1_analysis, h1_analysis, m15_analysis, config, avg_spread_price=cost.avg_spread_price
            )
            print(f"  {len(generated)} senales generadas")

            for g in generated:
                s = g.signal

                with_partial = simulate_managed_trade(
                    symbol=symbol, direction=s.direction, regime=s.regime, dominant_reason_kind=s.dominant_reason_kind,
                    signal_bar_idx=g.anchor_bar_idx, m15_df=candles["M15"], m15_analysis=m15_analysis,
                    original_sl=s.original_sl, partial_target=s.partial_target, final_target=s.final_target,
                    symbol_cost=cost, base_risk_pct=BASE_RISK_PCT, conviction_mult=s.conviction_multiplier,
                )
                if with_partial is not None:
                    with_partial_trades.append(with_partial)

                without_partial = simulate_managed_trade(
                    symbol=symbol, direction=s.direction, regime=s.regime, dominant_reason_kind=s.dominant_reason_kind,
                    signal_bar_idx=g.anchor_bar_idx, m15_df=candles["M15"], m15_analysis=m15_analysis,
                    original_sl=s.original_sl, partial_target=None, final_target=s.final_target,
                    symbol_cost=cost, base_risk_pct=BASE_RISK_PCT, conviction_mult=s.conviction_multiplier,
                )
                if without_partial is not None:
                    without_partial_trades.append(without_partial)
    finally:
        client.logout()

    print("\n=== CHEQUEO 3: PARCIAL EN PRIMER NIVEL INTERMEDIO -- CON vs. SIN ===\n")
    print("Con la regla de parcial (baseline):")
    print(" ", _fmt_metrics(compute_metrics(with_partial_trades)))
    print("Sin la regla de parcial (posicion completa a SL/TP final):")
    print(" ", _fmt_metrics(compute_metrics(without_partial_trades)))

    from collections import Counter
    print("\nDistribucion de exit_reason CON parcial:", Counter(t.exit_reason for t in with_partial_trades))
    print("Distribucion de exit_reason SIN parcial:", Counter(t.exit_reason for t in without_partial_trades))


if __name__ == "__main__":
    main()
