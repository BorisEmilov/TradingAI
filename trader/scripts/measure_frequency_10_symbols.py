"""Prompt "ampliar a 10 símbolos" (2026-09-25), punto 3: re-medir la frecuencia
real con los 7 pares nuevos verificados (`verify_and_fetch_new_fx_pairs.py`)
+ los 3 originales. Metodología IDÉNTICA a
`measure_pending_order_frequency_session_end.py` (se importan sus funciones,
no se copian): escaneo `require_retracement=False` -> dedup -> R:R neto de
costo >= 2.0 -> conflicto E1/E2 -> concurrencia por símbolo -> simulación de
llenado de la orden límite con expiración = fin de sesión de entrada.

Misma ventana: los CSV de los pares nuevos se bajaron el 2026-09-25 y se
RECORTAN al último timestamp de los CSV originales (2026-09-19), así los 10
símbolos cubren exactamente los mismos 3 meses. Sanity check: el subconjunto
de 3 símbolos originales debe reproducir 21 candidatos / 9 ejecutadas.
"""

from __future__ import annotations

import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import measure_pending_order_frequency_session_end as base
from trader.backtest.costs import cost_in_r, estimate_symbol_cost
from trader.config import load_config
from trader.mtf_strategies.session_risk import entry_session_end

ORIGINAL = ["EURUSD", "GBPUSD", "USDJPY"]
VERIFICATION_PATH = base.DATA_DIR / "new_fx_pairs_verification.json"
OUT_PATH = base.DATA_DIR.parent.parent / "logs" / "mtf_frequency_10_symbols.json"
N_PROCS = 2  # máquina con límite térmico -- ver memoria feedback_hardware_thermal_limits


def _common_end() -> pd.Timestamp:
    return min(base._load(s, "M15")["timestamp"].iloc[-1] for s in ORIGINAL)


def _load_clipped(symbol: str, tf: str, end: pd.Timestamp) -> pd.DataFrame:
    df = base._load(symbol, tf)
    return df[df["timestamp"] <= end].reset_index(drop=True)


def _scan_one(symbol: str, end: pd.Timestamp) -> list[dict]:
    load = partial(_load_clipped, end=end)
    return base._scan(load_config(), {symbol: load(symbol, "M15")}, load=load)


def main() -> None:
    with open(VERIFICATION_PATH) as f:
        new = json.load(f)["accepted"]
    symbols = ORIGINAL + new
    end = _common_end()
    print(f"{len(symbols)} símbolos: {symbols}\nfin común de datos: {end}")
    with open(base.DATA_DIR / "symbol_info_full.json") as f:
        symbol_info = json.load(f)

    m15_cache = {s: _load_clipped(s, "M15", end) for s in symbols}
    with Pool(N_PROCS) as pool:
        per_symbol = pool.map(partial(_scan_one, end=end), symbols)
    all_signals = [s for sigs in per_symbol for s in sigs]

    seen, unique = set(), []
    for s in all_signals:
        key = (s["symbol"], s["strategy"], s["setup"], s["direction"], s["sweep_timestamp"], s["mss_timestamp"], s["fvg_confirmed_at"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    sigs = pd.DataFrame(unique)
    sigs["generated_at"] = pd.to_datetime(sigs["generated_at"])
    sigs["fvg_confirmed_at"] = pd.to_datetime(sigs["fvg_confirmed_at"])

    cost = {s: estimate_symbol_cost(s, symbol_info[s]["point"], m15_cache[s]) for s in symbols}
    sigs["net_rr"] = [r["risk_reward"] - cost_in_r(cost[r["symbol"]], abs(r["entry"] - r["sl"])) for _, r in sigs.iterrows()]
    survived = sigs[sigs["net_rr"] >= base.MIN_NET_RR].reset_index(drop=True)

    result = base._run_conflict_concurrency_fill(
        survived, m15_cache, lambda row: entry_session_end(row["session"], pd.Timestamp(row["generated_at"]))
    )

    # mismo denominador que la medición de 3 símbolos (rango de generated_at de ESE subconjunto)
    orig = sigs[sigs["symbol"].isin(ORIGINAL)]
    n_days = (orig["generated_at"].max() - orig["generated_at"].min()).days or 1

    def count(key: str) -> pd.Series:
        return pd.Series([r["symbol"] for r in result[key]], dtype=object).value_counts()

    table = pd.DataFrame({
        "setups_unicos": sigs["symbol"].value_counts(),
        "rr_neto_ok": survived["symbol"].value_counts(),
        "conflicto": count("discarded_conflict"),
        "concurrencia": count("discarded_concurrency"),
        "expiradas": count("expired_unfilled"),
        "ejecutadas": count("executed"),
    }).reindex(symbols).fillna(0).astype(int)
    table["colocadas"] = table["expiradas"] + table["ejecutadas"]
    table["spread_pts_prom"] = [round(cost[s].avg_spread_points, 2) for s in symbols]
    table.loc["TOTAL_3_orig"] = table.loc[ORIGINAL].sum()
    table.loc["TOTAL_7_nuevos"] = table.loc[new].sum()
    table.loc["TOTAL_10"] = table.loc[symbols].sum()
    for label in ("TOTAL_3_orig", "TOTAL_7_nuevos", "TOTAL_10"):
        table.loc[label, "spread_pts_prom"] = float("nan")
    table["por_dia"] = (table["ejecutadas"] / n_days).round(3)
    table["por_semana"] = (table["ejecutadas"] / n_days * 7).round(2)
    print(f"\nventana: {n_days} días (mismo denominador que la medición de 3 símbolos)\n")
    print(table.to_string())

    by_strategy = pd.DataFrame(result["executed"]).groupby(["symbol", "strategy"]).size().unstack(fill_value=0) \
        if result["executed"] else pd.DataFrame()
    print(f"\nejecutadas por símbolo x estrategia:\n{by_strategy.to_string()}")

    with open(OUT_PATH, "w") as f:
        json.dump({"symbols": symbols, "common_end": str(end), "n_days": n_days,
                   "table": table.reset_index().to_dict(orient="records"), **result}, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
