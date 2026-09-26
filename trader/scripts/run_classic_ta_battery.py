"""prompt-inventario-real-instrumentos.md: run the approved 77-test classic-TA
battery (trend/breakout/volume/fibonacci/rsi), majors + metals, D1/swing
horizon. Reads ONLY from the local cache -- never touches the gateway.

Same pipeline discipline as run_quant_battery.py (the intraday battery):
EXPLORATION screen -> BH-FDR (q=0.05) + Bonferroni (alpha=0.05) over the full
77 -> VALIDATION on survivors with frozen params -> bootstrap (3 block sizes)
+ permutation -> economic check (spread + swap, swing_costs.py) + random
baseline for VALIDATION-survivors only. TEST is never read.

Indices excluded per explicit instruction (not tradable on this account, no
practical value in spending correction budget on them). Volume-spike
multiple=1.5 and RSI thresholds=30/70 were chosen by event-COUNT feasibility
only, before ever computing a return or p-value with any other candidate
value -- see the conversation record; not re-litigated here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from trader.quant import factors, stats
from trader.quant.splits import load_splits
from trader.quant.swing_costs import load_swap_info, swap_cost_frac

SYMBOL_INFO_PATH = Path(__file__).resolve().parent.parent / "data" / "quant_battery" / "symbol_info_full.json"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"]
PRIMARY_N_PERM = 500
CONFIRM_N_PERM = 2000
BOOT_BLOCK_SIZES_EVENTS = (10, 20, 40)
ALPHA = 0.05
FDR_Q = 0.05
HORIZON = 10  # trading days -- shared by breakout/volume/fibonacci/rsi; trend uses 21 (its own literature-standard horizon)
TREND_HORIZON = 21


@dataclass
class TestOutcome:
    label: str
    family: str
    symbol: str
    param: str
    n_events: int
    observed_effect: float
    perm_block_bars: int
    p_value: float
    fdr_reject: bool = False
    bonferroni_reject: bool = False
    validation_n_events: int | None = None
    validation_effect: float | None = None
    validation_same_sign: bool | None = None
    validation_p_value: float | None = None
    validation_boot: list | None = None
    validation_boot_all_exclude_zero: bool | None = None
    horizon_days: int = HORIZON
    cost_frac: float | None = None
    net_effect_validation: float | None = None
    random_baseline_percentile: float | None = None


def _perm_block_bars(base: int) -> int:
    return max(20, base)


def _build_battery() -> list[dict]:
    tests = []
    for fast, slow in [(10, 50), (20, 100), (50, 200)]:
        for symbol in SYMBOLS:
            tests.append({"family": "trend", "symbol": symbol, "fast": fast, "slow": slow})
    for n in (20, 55):
        for symbol in SYMBOLS:
            tests.append({"family": "breakout", "symbol": symbol, "n": n})
    for symbol in SYMBOLS:
        tests.append({"family": "volume", "symbol": symbol, "multiple": 1.5})
    for level in (0.382, 0.5, 0.618):
        for symbol in SYMBOLS:
            tests.append({"family": "fibonacci", "symbol": symbol, "level": level})
    for period in (14, 21):
        for symbol in SYMBOLS:
            tests.append({"family": "rsi", "symbol": symbol, "period": period, "oversold": 30, "overbought": 70})
    return tests


def _run_factor(spec: dict, candles: pd.DataFrame):
    family = spec["family"]
    if family == "trend":
        return factors.trend_ma_cross_factor(candles, spec["fast"], spec["slow"], TREND_HORIZON)
    if family == "breakout":
        return factors.channel_breakout_factor(candles, spec["n"], HORIZON)
    if family == "volume":
        return factors.volume_spike_factor(candles, spec["multiple"], HORIZON)
    if family == "fibonacci":
        return factors.fibonacci_reaction_factor(candles, "D1", spec["level"], HORIZON)
    if family == "rsi":
        return factors.rsi_reversal_factor(candles, spec["period"], spec["oversold"], spec["overbought"], HORIZON)
    raise ValueError(family)


def _label(spec: dict) -> str:
    family, symbol = spec["family"], spec["symbol"]
    if family == "trend":
        return f"trend_MA{spec['fast']}-{spec['slow']}_{symbol}"
    if family == "breakout":
        return f"breakout_N{spec['n']}_{symbol}"
    if family == "volume":
        return f"volume_spike{spec['multiple']}_{symbol}"
    if family == "fibonacci":
        return f"fibonacci_{spec['level']}_{symbol}"
    if family == "rsi":
        return f"rsi{spec['period']}_{symbol}"
    raise ValueError(family)


def _compute_explore(spec: dict) -> TestOutcome:
    symbol, family = spec["symbol"], spec["family"]
    split = load_splits(symbol, "D1")
    r = _run_factor(spec, split.exploration)
    label = _label(spec)
    horizon = TREND_HORIZON if family == "trend" else HORIZON

    if family == "trend":
        if r.n_obs == 0:
            return TestOutcome(label, family, symbol, "", 0, 0.0, _perm_block_bars(horizon), 1.0, horizon_days=horizon)
        block_bars = _perm_block_bars(horizon)
        perm = stats.block_permutation_pvalue_dense_signal(r.direction, r.forward_returns, block_bars, n_perm=PRIMARY_N_PERM, seed=1)
        observed = float((r.direction * r.forward_returns).mean())
        return TestOutcome(label, family, symbol, f"{spec['fast']}/{spec['slow']}", r.n_obs, observed, block_bars, perm.p_value, horizon_days=horizon)

    if r.n_events == 0:
        return TestOutcome(label, family, symbol, "", 0, 0.0, _perm_block_bars(horizon), 1.0, horizon_days=horizon)
    block_bars = _perm_block_bars(horizon)
    perm = stats.block_permutation_pvalue_sparse_event(r.full_returns, r.event_mask, block_bars, n_perm=PRIMARY_N_PERM, seed=1)
    param = {"breakout": f"N={spec.get('n')}", "volume": f"x{spec.get('multiple')}", "fibonacci": f"{spec.get('level')}", "rsi": f"period={spec.get('period')}"}[family]
    return TestOutcome(label, family, symbol, param, r.n_events, perm.observed, block_bars, perm.p_value, horizon_days=horizon)


def _validate_survivor(spec: dict, outcome: TestOutcome) -> None:
    symbol, family = spec["symbol"], spec["family"]
    split = load_splits(symbol, "D1")
    r = _run_factor(spec, split.validation)

    if family == "trend":
        if r.n_obs == 0:
            outcome.validation_n_events = 0
            return
        val_effect = float((r.direction * r.forward_returns).mean())
        outcome.validation_n_events = r.n_obs
        outcome.validation_effect = val_effect
        outcome.validation_same_sign = np.sign(val_effect) == np.sign(outcome.observed_effect)
        perm = stats.block_permutation_pvalue_dense_signal(r.direction, r.forward_returns, outcome.perm_block_bars, n_perm=CONFIRM_N_PERM, seed=2)
        outcome.validation_p_value = perm.p_value
        event_series = r.direction * r.forward_returns
    else:
        if r.n_events == 0:
            outcome.validation_n_events = 0
            return
        val_effect = float(r.full_returns[r.event_mask].mean())
        outcome.validation_n_events = r.n_events
        outcome.validation_effect = val_effect
        outcome.validation_same_sign = np.sign(val_effect) == np.sign(outcome.observed_effect)
        perm = stats.block_permutation_pvalue_sparse_event(r.full_returns, r.event_mask, outcome.perm_block_bars, n_perm=CONFIRM_N_PERM, seed=2)
        outcome.validation_p_value = perm.p_value
        event_series = r.full_returns[r.event_mask]

    boot_results = [stats.block_bootstrap_mean_ci(event_series, bs, n_boot=CONFIRM_N_PERM, seed=3) for bs in BOOT_BLOCK_SIZES_EVENTS]
    outcome.validation_boot = boot_results
    outcome.validation_boot_all_exclude_zero = all(b.excludes_zero for b in boot_results)


def _economic_check(spec: dict, outcome: TestOutcome, symbol_info: dict) -> None:
    symbol = spec["symbol"]
    m15 = load_splits(symbol, "M15")
    full_m15 = pd.concat([m15.exploration, m15.validation], ignore_index=True)
    avg_spread_points = float(full_m15["spread"].mean()) if "spread" in full_m15.columns else 0.0
    si = symbol_info[symbol]
    point = si["point"]
    avg_spread_price = avg_spread_points * point

    split = load_splits(symbol, "D1")
    validation = split.validation
    mean_price = float(validation["close"].mean())
    spread_cost_frac = avg_spread_price / mean_price

    swap = load_swap_info(symbol, si)
    entry = pd.Timestamp("2024-01-01", tz="UTC")  # representative anchor, only the SPAN in calendar days matters
    exit_ = entry + pd.Timedelta(days=round(outcome.horizon_days * 7 / 5))  # trading days -> approx calendar days
    # sparse/dense factors here fire both long and short bets -- use whichever
    # side (long/short) costs MORE as a conservative bound, not an optimistic
    # pick (this account's swap_long/swap_short differ a lot per symbol, e.g.
    # GBPUSD short is ~11x its long rate).
    swap_frac = max(
        swap_cost_frac("long", entry, exit_, swap, mean_price),
        swap_cost_frac("short", entry, exit_, swap, mean_price),
    )

    outcome.cost_frac = spread_cost_frac + swap_frac
    outcome.net_effect_validation = outcome.validation_effect - outcome.cost_frac

    rng = np.random.default_rng(4)
    r = _run_factor(spec, validation)
    n_events = outcome.validation_n_events
    if spec["family"] == "trend":
        pool = r.forward_returns
        n_pool = len(pool)
        trial_means = np.empty(2000)
        for i in range(2000):
            rand_dir = rng.choice([-1.0, 1.0], size=n_pool)
            trial_means[i] = (rand_dir * pool).mean()
    else:
        pool = r.full_returns
        n_pool = len(pool)
        trial_means = np.empty(2000)
        for i in range(2000):
            idx = rng.choice(n_pool, size=min(n_events, n_pool), replace=False)
            trial_means[i] = pool[idx].mean()

    outcome.random_baseline_percentile = float((trial_means < outcome.validation_effect).mean() * 100)


def main() -> None:
    with open(SYMBOL_INFO_PATH) as f:
        symbol_info = json.load(f)

    battery = _build_battery()
    assert len(battery) == 77, f"expected 77 tests, built {len(battery)}"

    print(f"=== EXPLORATION: computing {len(battery)} tests ===")
    outcomes = [_compute_explore(spec) for spec in battery]

    p_values = [o.p_value for o in outcomes]
    fdr = stats.benjamini_hochberg(p_values, q=FDR_Q)
    bonf = stats.bonferroni(p_values, alpha=ALPHA)
    for o, f, b in zip(outcomes, fdr, bonf):
        o.fdr_reject = f
        o.bonferroni_reject = b

    print("\n=== TABLA COMPLETA (77 tests, EXPLORATION), ordenada por p ===")
    print(f"{'label':<28} {'param':<12} {'n':>6} {'effect':>10} {'p':>8} {'FDR':>5} {'Bonf':>5}")
    for o in sorted(outcomes, key=lambda o: o.p_value):
        print(f"{o.label:<28} {o.param:<12} {o.n_events:>6} {o.observed_effect:>10.6f} {o.p_value:>8.4f} {'YES' if o.fdr_reject else '-':>5} {'YES' if o.bonferroni_reject else '-':>5}")

    survivors = [o for o, spec in zip(outcomes, battery) if o.fdr_reject]
    survivor_specs = [spec for o, spec in zip(outcomes, battery) if o.fdr_reject]
    print(f"\n=== SOBREVIVIENTES FDR (q={FDR_Q}): {len(survivors)} de {len(outcomes)} ===")
    for o in survivors:
        print(f"  {o.label} ({o.param}): p={o.p_value:.5f}, effect={o.observed_effect:.6f}, n={o.n_events}")

    print("\n=== VALIDATION (solo sobrevivientes, parametros congelados) ===")
    for o, spec in zip(survivors, survivor_specs):
        _validate_survivor(spec, o)
        boot_str = ", ".join(f"bs={b.block_size}:[{b.ci_lo:.6f},{b.ci_hi:.6f}]" for b in (o.validation_boot or []))
        print(f"  {o.label}: n={o.validation_n_events}, effect={o.validation_effect}, same_sign={o.validation_same_sign}, "
              f"perm_p={o.validation_p_value}, boot_excludes_zero_all={o.validation_boot_all_exclude_zero} ({boot_str})")

    val_survivors = [
        (o, spec) for o, spec in zip(survivors, survivor_specs)
        if o.validation_same_sign and o.validation_boot_all_exclude_zero
    ]
    print(f"\n=== SOBREVIVIENTES VALIDATION: {len(val_survivors)} ===")

    print("\n=== CHEQUEO ECONOMICO (spread + swap) + LINEA BASE ALEATORIA ===")
    for o, spec in val_survivors:
        _economic_check(spec, o, symbol_info)
        print(f"  {o.label}: gross={o.validation_effect:.6f}, cost(spread+swap)={o.cost_frac:.6f}, net={o.net_effect_validation:.6f}, "
              f"random_baseline_percentile={o.random_baseline_percentile:.1f}%")

    print("\n=== RESUMEN FINAL ===")
    print(f"Tests totales: {len(outcomes)}")
    print(f"Sobreviven FDR (q={FDR_Q}): {len(survivors)}")
    print(f"Sobreviven Bonferroni (alpha={ALPHA}): {sum(o.bonferroni_reject for o in outcomes)}")
    print(f"Sobreviven VALIDATION: {len(val_survivors)}")
    econ_viable = [o for o, _ in val_survivors if o.net_effect_validation is not None and o.net_effect_validation > 0]
    print(f"Netos positivos tras coste real: {len(econ_viable)}")
    print("Indices: EXCLUIDOS (no operables). TEST: NO LEIDO.")


if __name__ == "__main__":
    main()
