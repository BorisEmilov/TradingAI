"""Parameter sweep requested after the Fase 4 backtest found only 1 trade in
~2 years across 3 symbols: vary the filters that gate a signal and measure
the frequency/quality trade-off, instead of guessing new values.

Fixed for every combination (per instruction): min_risk_reward=2.0,
confluence.min_timeframes=2. Detectors are computed ONCE per symbol (none of
the swept parameters affect detector output, only decision logic in
`run_from_analyses`), then every combination re-runs cheaply against that
same precomputed data -- see trader/backtest/engine.py's
`run_symbol_backtest_from_analyses` docstring.

Two things asked for in the sweep turned out to already be at the value
being "proposed", given the actual code (not guessed -- verified against
config.py/pipeline/engine.py before writing this script):
  - fvg.min_gap_pct is already 0.0 (loosest possible: ANY non-overlapping
    gap counts). There is no more permissive value to sweep to.
  - The session gate already accepts ANY of Asia/London/NY active, not just
    the London-NY overlap -- "London full + NY full, no overlap required" is
    the CURRENT default (require_overlap_or_killzone=False), not a
    relaxation. A stricter "overlap/killzone only" variant is included below
    as the reference point the prompt's framing implied was current.

Usage: python scripts/run_sensitivity_sweep.py [--count 50000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.engine import run_symbol_backtest_from_analyses
from trader.backtest.metrics import compute_metrics
from trader.config import TraderConfig, load_config
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.engine import TimeframeAnalysis

TIMEFRAMES = ("D1", "H1", "M15")  # H4 removed from the pipeline, see prompt-eliminar-gate-d1-h4.md

# M15 (~2 years available) is the binding constraint on how far back a signal
# can be anchored -- D1/H1 only need to cover that same window plus some
# margin for warm-up, not the 33-56 years of D1 history the account actually
# has. Requesting all of it (as an earlier version of this script did,
# uniformly, with count=50000 for every timeframe) turned out to be
# unreliable: 3 straight 502/503s fetching 50000 D1 bars right after login,
# even with retries -- a much smaller, still-generous request is both lighter
# and apparently what the gateway/terminal actually handles reliably.
DEFAULT_COUNTS = {"D1": 1500, "H1": 20000, "M15": 50000}

POI_TOLERANCE_VALUES = [0.1, 0.15, 0.2]  # % ; 0.1 = current baseline
MIN_CONFLUENCES_VALUES = [3, 2]  # 3 = current baseline
REQUIRE_OVERLAP_VALUES = [False, True]  # False = current baseline (any active session)


def _variant(base: TraderConfig, *, poi_tolerance_pct: float, min_confluences: int, require_overlap: bool) -> TraderConfig:
    return dataclasses.replace(
        base,
        poi=dataclasses.replace(base.poi, tolerance_pct=poi_tolerance_pct),
        confluence=dataclasses.replace(base.confluence, min_confluences=min_confluences, min_timeframes=2),
        sessions=dataclasses.replace(base.sessions, require_overlap_or_killzone=require_overlap),
        risk=dataclasses.replace(base.risk, min_risk_reward=2.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None, help="override: same count for every timeframe")
    args = parser.parse_args()

    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    per_symbol_data = {}
    try:
        for symbol in config.symbols:
            print(f"=== {symbol}: fetching history + running detectors (once) ===")
            candles = {
                tf: client.candles(symbol, tf, args.count or DEFAULT_COUNTS[tf], timeout=120) for tf in TIMEFRAMES
            }
            analyses = {tf: TimeframeAnalysis.from_candles(candles[tf], tf, config) for tf in TIMEFRAMES}
            info = client.symbol_info(symbol, timeout=30)
            cost = estimate_symbol_cost(symbol, float(info["point"]), candles["M15"])
            per_symbol_data[symbol] = (analyses, candles["M15"], cost)
    finally:
        client.logout()

    combos = []
    for tol in POI_TOLERANCE_VALUES:
        for mc in MIN_CONFLUENCES_VALUES:
            for ro in REQUIRE_OVERLAP_VALUES:
                combos.append((tol, mc, ro))

    print(f"\n=== corriendo {len(combos)} combinaciones sobre {len(config.symbols)} simbolos ===\n")

    results = []
    for tol, mc, ro in combos:
        variant_config = _variant(config, poi_tolerance_pct=tol, min_confluences=mc, require_overlap=ro)

        all_trades = []
        all_no_signals = []
        for symbol, (analyses, m15_df, cost) in per_symbol_data.items():
            r = run_symbol_backtest_from_analyses(symbol, analyses, m15_df, variant_config, cost)
            all_trades.extend(r.trades)
            all_no_signals.extend(r.no_signals)

        signals_before_rr = len(all_trades) + sum(
            1 for n in all_no_signals if n.stage == "risk" and n.reason == "risk_reward_below_minimum"
        )
        m = compute_metrics(all_trades)

        results.append(
            {
                "poi_tolerance_pct": tol,
                "min_confluences": mc,
                "require_overlap_or_killzone": ro,
                "signals_before_rr_filter": signals_before_rr,
                "trades_final": m.n_trades,
                "win_rate": m.win_rate,
                "expectancy_r": m.expectancy_r,
                "profit_factor": m.profit_factor if m.profit_factor != float("inf") else None,
                "max_drawdown_r": m.max_drawdown_r,
            }
        )
        pf_str = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        print(
            f"tol={tol:.2f}% min_conf={mc} overlap_only={ro!s:5} -> "
            f"señales_pre_RR={signals_before_rr:4d} trades={m.n_trades:3d} "
            f"expectancy_r={m.expectancy_r:+.3f} pf={pf_str} max_dd_r={m.max_drawdown_r:.2f}"
        )

    logs_dir = Path(config.signals_log_path).parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "sensitivity_sweep.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
