"""prompt-fase-cuantitativa.md: run the approved 60-test quant-factor battery.

Reads ONLY from the local cache written by fetch_quant_battery_data.py --
never touches the gateway. Pipeline, per the approved design:

  1. EXPLORATION: compute all 60 tests, primary permutation p-value each
     (single pre-registered block size per test -- no block-size fishing at
     this stage), apply BH-FDR (q=0.05, primary) and Bonferroni (alpha=0.05,
     reported alongside) across the full set of 60.
  2. Survivors only: re-run on VALIDATION with FROZEN parameters (same N;
     for session/weekday, the sign is whatever EXPLORATION's empirical
     direction was -- never re-picked on VALIDATION), check same sign +
     3-block-size bootstrap CI + permutation p-value.
  3. VALIDATION-survivors only (same sign, bootstrap excludes zero in the
     primary block size): economic check (net of real average spread cost,
     from the cached M15 spread column + cached symbol point size) and a
     random-entry baseline comparison.

TEST is never read by this script -- `trader.quant.splits.load_splits`
returns it, but nothing here touches `.test`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from trader.config import load_config
from trader.quant import factors, stats
from trader.quant.splits import load_splits

SYMBOL_INFO_PATH = Path(__file__).resolve().parent.parent / "data" / "quant_battery" / "symbol_info.json"
PRIMARY_N_PERM = 500
CONFIRM_N_PERM = 2000
BOOT_BLOCK_SIZES_EVENTS = (10, 20, 40)
ALPHA = 0.05
FDR_Q = 0.05


@dataclass
class TestOutcome:
    label: str
    family: str
    symbol: str
    timeframe: str
    param: str
    n_events: int
    observed_effect: float
    perm_block_bars: int
    p_value: float
    fdr_reject: bool = False
    bonferroni_reject: bool = False
    # validation stage (filled only for survivors)
    validation_n_events: int | None = None
    validation_effect: float | None = None
    validation_same_sign: bool | None = None
    validation_p_value: float | None = None
    validation_boot: list | None = None  # list of stats.BootstrapResult
    validation_boot_all_exclude_zero: bool | None = None
    # economic stage (filled only for validation-survivors)
    cost_frac: float | None = None
    net_effect_validation: float | None = None
    random_baseline_percentile: float | None = None


def _perm_block_bars(n: int | None, floor: int = 20) -> int:
    return max(floor, 4 * n) if n else floor


def _build_battery() -> list[dict]:
    tests = []
    for tf, ns in factors.MOMENTUM_REVERSION_N.items():
        for n in ns:
            for symbol in SYMBOLS:
                tests.append({"family": "momentum", "symbol": symbol, "timeframe": tf, "n": n})
                tests.append({"family": "mean_reversion", "symbol": symbol, "timeframe": tf, "n": n})
    for session in ("asia", "london", "new_york"):
        for symbol in SYMBOLS:
            tests.append({"family": "session", "symbol": symbol, "timeframe": "M15", "session": session})
    for wd, wd_name in enumerate(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")):
        for symbol in SYMBOLS:
            tests.append({"family": "weekday", "symbol": symbol, "timeframe": "D1", "weekday": wd, "weekday_name": wd_name})
    return tests


def _compute_explore(spec: dict, session_labels_cache: dict) -> TestOutcome:
    symbol, tf, family = spec["symbol"], spec["timeframe"], spec["family"]
    split = load_splits(symbol, tf)
    explore = split.exploration

    if family == "momentum":
        n = spec["n"]
        r = factors.momentum_factor(explore, n)
        block_bars = _perm_block_bars(n)
        perm = stats.block_permutation_pvalue_dense_signal(r.direction, r.forward_returns, block_bars, n_perm=PRIMARY_N_PERM, seed=1)
        observed = float((r.direction * r.forward_returns).mean())
        label = f"momentum_{tf}_N{n}_{symbol}"
        param = f"N={n}"
        n_events = r.n_obs
    elif family == "mean_reversion":
        n = spec["n"]
        r = factors.mean_reversion_factor(explore, n)
        block_bars = _perm_block_bars(n)
        if r.n_events == 0:
            return TestOutcome(f"mean_reversion_{tf}_N{n}_{symbol}", family, symbol, tf, f"N={n}", 0, 0.0, block_bars, 1.0)
        perm = stats.block_permutation_pvalue_sparse_event(r.full_returns, r.event_mask, block_bars, n_perm=PRIMARY_N_PERM, seed=1)
        observed = perm.observed
        label = f"mean_reversion_{tf}_N{n}_{symbol}"
        param = f"N={n}"
        n_events = r.n_events
    elif family == "session":
        session = spec["session"]
        if symbol not in session_labels_cache:
            config = load_config()
            session_labels_cache[symbol] = factors.compute_session_labels(explore, config.sessions)
        labels = session_labels_cache[symbol]
        r = factors.session_factor(explore, labels, session)
        block_bars = _perm_block_bars(None, floor=20)
        if r.n_events == 0:
            return TestOutcome(f"session_{session}_{symbol}", family, symbol, tf, session, 0, 0.0, block_bars, 1.0)
        perm = stats.block_permutation_pvalue_sparse_event(r.full_returns, r.event_mask, block_bars, n_perm=PRIMARY_N_PERM, seed=1)
        observed = perm.observed
        label = f"session_{session}_{symbol}"
        param = session
        n_events = r.n_events
    elif family == "weekday":
        wd, wd_name = spec["weekday"], spec["weekday_name"]
        r = factors.weekday_factor(explore, wd)
        block_bars = _perm_block_bars(None, floor=10)
        if r.n_events == 0:
            return TestOutcome(f"weekday_{wd_name}_{symbol}", family, symbol, tf, wd_name, 0, 0.0, block_bars, 1.0)
        perm = stats.block_permutation_pvalue_sparse_event(r.full_returns, r.event_mask, block_bars, n_perm=PRIMARY_N_PERM, seed=1)
        observed = perm.observed
        label = f"weekday_{wd_name}_{symbol}"
        param = wd_name
        n_events = r.n_events
    else:
        raise ValueError(family)

    return TestOutcome(label, family, symbol, tf, param, n_events, observed, block_bars, perm.p_value)


def _recompute_signal(spec: dict, candles: pd.DataFrame, sign_hat: float | None, session_labels: pd.DataFrame | None):
    family = spec["family"]
    if family == "momentum":
        r = factors.momentum_factor(candles, spec["n"])
        return r.direction, r.forward_returns, None
    if family == "mean_reversion":
        r = factors.mean_reversion_factor(candles, spec["n"])
        return None, None, r
    if family == "session":
        r = factors.session_factor(candles, session_labels, spec["session"])
        return None, None, r
    if family == "weekday":
        r = factors.weekday_factor(candles, spec["weekday"])
        return None, None, r
    raise ValueError(family)


def _validate_survivor(spec: dict, outcome: TestOutcome) -> None:
    symbol, tf, family = spec["symbol"], spec["timeframe"], spec["family"]
    split = load_splits(symbol, tf)
    validation = split.validation

    session_labels = None
    if family == "session":
        config = load_config()
        session_labels = factors.compute_session_labels(validation, config.sessions)

    direction, forward, sparse = _recompute_signal(spec, validation, None, session_labels)

    if family == "momentum":
        if len(direction) == 0:
            outcome.validation_n_events = 0
            return
        val_effect = float((direction * forward).mean())
        outcome.validation_n_events = len(direction)
        outcome.validation_effect = val_effect
        outcome.validation_same_sign = np.sign(val_effect) == np.sign(outcome.observed_effect)
        perm = stats.block_permutation_pvalue_dense_signal(direction, forward, outcome.perm_block_bars, n_perm=CONFIRM_N_PERM, seed=2)
        outcome.validation_p_value = perm.p_value
        event_series = direction * forward  # "event" series for bootstrap = the whole aligned series (dense)
    else:
        if sparse is None or sparse.n_events == 0:
            outcome.validation_n_events = 0 if sparse is None else sparse.n_events
            return
        val_effect = float(sparse.full_returns[sparse.event_mask].mean())
        outcome.validation_n_events = sparse.n_events
        outcome.validation_effect = val_effect
        outcome.validation_same_sign = np.sign(val_effect) == np.sign(outcome.observed_effect)
        perm = stats.block_permutation_pvalue_sparse_event(sparse.full_returns, sparse.event_mask, outcome.perm_block_bars, n_perm=CONFIRM_N_PERM, seed=2)
        outcome.validation_p_value = perm.p_value
        event_series = sparse.full_returns[sparse.event_mask]

    boot_results = [stats.block_bootstrap_mean_ci(event_series, bs, n_boot=CONFIRM_N_PERM, seed=3) for bs in BOOT_BLOCK_SIZES_EVENTS]
    outcome.validation_boot = boot_results
    outcome.validation_boot_all_exclude_zero = all(b.excludes_zero for b in boot_results)


def _economic_check(spec: dict, outcome: TestOutcome, symbol_info: dict) -> None:
    symbol, tf = spec["symbol"], spec["timeframe"]
    m15 = load_splits(symbol, "M15")
    full_m15 = pd.concat([m15.exploration, m15.validation], ignore_index=True)
    avg_spread_points = float(full_m15["spread"].mean()) if "spread" in full_m15.columns else 0.0
    point = symbol_info[symbol]["point"]
    avg_spread_price = avg_spread_points * point
    mean_price = float(full_m15["close"].mean())
    cost_frac = avg_spread_price / mean_price  # round-trip spread as a fraction of price, log-return-comparable

    outcome.cost_frac = cost_frac
    outcome.net_effect_validation = outcome.validation_effect - cost_frac

    # random-entry baseline on the SAME validation series
    split = load_splits(symbol, tf)
    validation = split.validation
    rng = np.random.default_rng(4)
    n_events = outcome.validation_n_events
    session_labels = None
    if spec["family"] == "session":
        config = load_config()
        session_labels = factors.compute_session_labels(validation, config.sessions)
    direction, forward, sparse = _recompute_signal(spec, validation, None, session_labels)

    if spec["family"] == "momentum":
        pool = forward  # random direction baseline: iid random sign vs the real forward-return pool
        n_pool = len(pool)
        trial_means = np.empty(2000)
        for i in range(2000):
            rand_dir = rng.choice([-1.0, 1.0], size=n_pool)
            trial_means[i] = (rand_dir * pool).mean()
    else:
        pool = sparse.full_returns
        n_pool = len(pool)
        trial_means = np.empty(2000)
        for i in range(2000):
            idx = rng.choice(n_pool, size=min(n_events, n_pool), replace=False)
            trial_means[i] = pool[idx].mean()

    percentile = float((trial_means < outcome.validation_effect).mean() * 100)
    outcome.random_baseline_percentile = percentile


SYMBOLS: list[str] = []


def main() -> None:
    global SYMBOLS
    config = load_config()
    SYMBOLS = config.symbols

    with open(SYMBOL_INFO_PATH) as f:
        symbol_info = json.load(f)

    battery = _build_battery()
    assert len(battery) == 60, f"expected 60 tests, built {len(battery)}"

    print(f"=== EXPLORATION: computing {len(battery)} tests ===")
    session_labels_cache: dict = {}
    outcomes = [_compute_explore(spec, session_labels_cache) for spec in battery]

    p_values = [o.p_value for o in outcomes]
    fdr = stats.benjamini_hochberg(p_values, q=FDR_Q)
    bonf = stats.bonferroni(p_values, alpha=ALPHA)
    for o, f, b in zip(outcomes, fdr, bonf):
        o.fdr_reject = f
        o.bonferroni_reject = b

    print("\n=== TABLA COMPLETA (60 tests, EXPLORATION) ===")
    print(f"{'label':<32} {'n':>7} {'effect':>10} {'p':>8} {'FDR':>5} {'Bonf':>5}")
    for o in sorted(outcomes, key=lambda o: o.p_value):
        print(f"{o.label:<32} {o.n_events:>7} {o.observed_effect:>10.6f} {o.p_value:>8.4f} {'YES' if o.fdr_reject else '-':>5} {'YES' if o.bonferroni_reject else '-':>5}")

    survivors = [o for o, spec in zip(outcomes, battery) if o.fdr_reject]
    survivor_specs = [spec for o, spec in zip(outcomes, battery) if o.fdr_reject]
    print(f"\n=== SOBREVIVIENTES FDR (q={FDR_Q}): {len(survivors)} de {len(outcomes)} ===")
    for o in survivors:
        print(f"  {o.label}: p={o.p_value:.4f}, effect={o.observed_effect:.6f}, n={o.n_events}")

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
    print(f"\n=== SOBREVIVIENTES VALIDATION (mismo signo + bootstrap excluye cero en los 3 tamanos): {len(val_survivors)} ===")

    print("\n=== CHEQUEO ECONOMICO + LINEA BASE ALEATORIA (solo sobrevivientes de VALIDATION) ===")
    for o, spec in val_survivors:
        _economic_check(spec, o, symbol_info)
        print(f"  {o.label}: gross={o.validation_effect:.6f}, cost={o.cost_frac:.6f}, net={o.net_effect_validation:.6f}, "
              f"random_baseline_percentile_of_observed={o.random_baseline_percentile:.1f}%")

    print("\n=== RESUMEN FINAL ===")
    print(f"Tests totales: {len(outcomes)}")
    print(f"Sobreviven FDR (q={FDR_Q}): {len(survivors)}")
    print(f"Sobreviven Bonferroni (alpha={ALPHA}): {sum(o.bonferroni_reject for o in outcomes)}")
    print(f"Sobreviven VALIDATION (mismo signo + bootstrap robusto): {len(val_survivors)}")
    economically_viable = [o for o, _ in val_survivors if o.net_effect_validation is not None and o.net_effect_validation > 0]
    print(f"Netos positivos tras coste real: {len(economically_viable)}")
    print("TEST: NO LEIDO (por diseno).")


if __name__ == "__main__":
    main()
