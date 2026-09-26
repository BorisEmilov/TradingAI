"""prompt-3-chequeos-antes-de-cerrar.md, chequeo 2: test de senal invertida.

Para cada senal generada por las 4 capas (con el piso absoluto del chequeo 1
ya aplicado), simula DOS variantes desde el mismo punto de entrada: la
direccion normal (baseline) y la direccion INVERTIDA -- SL y TP final
reflejados sobre el precio de entrada preservando las mismas magnitudes de
riesgo/recompensa, misma gestion estructural, mismo sizing. Si "peor que el
azar" fuera un bug de signo (confirmacion detectando agotamiento cuando en
realidad hay continuacion, o viceversa), la version invertida deberia dar
expectancy positivo y significativo. Si sigue siendo neutro/negativo,
confirma ruido genuino sin direccion explotable.
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


def _favorable_side(entry: float, distance: float, direction: str) -> float:
    """TP/parcial: el lado favorable esta ARRIBA del entry para long, ABAJO para short."""
    return entry + distance if direction == "long" else entry - distance


def _unfavorable_side(entry: float, distance: float, direction: str) -> float:
    """SL: el lado desfavorable esta ABAJO del entry para long, ARRIBA para short --
    signo OPUESTO al de TP para la MISMA direccion. Un primer intento de este script
    uso la misma formula para ambos (bug encontrado al ver n=1 de 4491 invertidas
    sobrevivir el chequeo de validez -- SL terminaba del lado equivocado del entry
    para la direccion invertida, rechazando casi todo por risk_price<=0)."""
    return entry - distance if direction == "long" else entry + distance


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

    normal_trades = []
    inverted_trades = []

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
                entry = s.entry_reference_price

                normal = simulate_managed_trade(
                    symbol=symbol, direction=s.direction, regime=s.regime, dominant_reason_kind=s.dominant_reason_kind,
                    signal_bar_idx=g.anchor_bar_idx, m15_df=candles["M15"], m15_analysis=m15_analysis,
                    original_sl=s.original_sl, partial_target=s.partial_target, final_target=s.final_target,
                    symbol_cost=cost, base_risk_pct=BASE_RISK_PCT, conviction_mult=s.conviction_multiplier,
                )
                if normal is not None:
                    normal_trades.append(normal)

                inv_direction = "short" if s.direction == "long" else "long"
                risk_distance = abs(s.original_sl - entry)
                reward_distance = abs(s.final_target - entry)
                inv_sl = _unfavorable_side(entry, risk_distance, inv_direction)
                inv_final_target = _favorable_side(entry, reward_distance, inv_direction)
                inv_partial = (
                    _favorable_side(entry, abs(s.partial_target - entry), inv_direction)
                    if s.partial_target is not None
                    else None
                )

                inverted = simulate_managed_trade(
                    symbol=symbol, direction=inv_direction, regime=s.regime, dominant_reason_kind=s.dominant_reason_kind,
                    signal_bar_idx=g.anchor_bar_idx, m15_df=candles["M15"], m15_analysis=m15_analysis,
                    original_sl=inv_sl, partial_target=inv_partial, final_target=inv_final_target,
                    symbol_cost=cost, base_risk_pct=BASE_RISK_PCT, conviction_mult=s.conviction_multiplier,
                )
                if inverted is not None:
                    inverted_trades.append(inverted)
    finally:
        client.logout()

    print("\n=== CHEQUEO 2: SENAL INVERTIDA ===\n")
    print("Normal (baseline, direccion tal como la produce el sistema):")
    print(" ", _fmt_metrics(compute_metrics(normal_trades)))
    print("Invertida (SL/TP reflejados, misma magnitud de riesgo/recompensa, direccion opuesta):")
    print(" ", _fmt_metrics(compute_metrics(inverted_trades)))


if __name__ == "__main__":
    main()
