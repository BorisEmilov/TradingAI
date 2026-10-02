"""Prueba fuera de muestra PRE-REGISTRADA (2026-10-02). Ver
logs/oos_preregistro_2026-10-02.md -- ese documento manda; este script solo
lo ejecuta. Un solo uso para la ventana OOS.

Pipeline = el mismo de las mediciones previas, sin ningún parámetro nuevo:
escaneo causal (`measure_frequency_10_symbols._scan_one` ->
`measure_pending_order_frequency_session_end._scan`, require_retracement=False)
-> dedup -> R:R neto de costo >= 2.0 -> `dl._simulate` (conflicto E1/E2,
concurrencia 1/símbolo, corte diario -1.5R, orden límite al 50% FVG con
chequeo de precio y expiración fin de sesión, parcial 50% TP1 + BE, TP2, sin
salida temporal) -> métricas de `honest_reevaluation` (bootstrap por bloques
diarios, escenarios pesimistas).

  --window insample  valida el harness: debe reproducir 69 candidatos, n=34,
                     +0.474R y pesimista +0.330R (ventana de diseño, ya vista).
  --window oos       LA corrida única. No correr sin el pre-registro confirmado.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import honest_reevaluation as hr
import measure_frequency_10_symbols as ten
import measure_pending_order_frequency_session_end as base
from trader.backtest.costs import cost_in_r, estimate_symbol_cost

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY"]
WINDOWS = {
    # ventana de diseño: idéntica a logs/mtf_frequency_10_symbols.json (fin común 2026-09-25 16:15, 3 meses)
    "insample": {"scan_end": pd.Timestamp("2026-09-25 16:15", tz="UTC"), "months_back": 3,
                 "start": None, "resolve_end": pd.Timestamp("2026-09-25 16:15", tz="UTC")},
    # OOS pre-registrada: señales generadas en [2026-01-01, 2026-06-19]; 2026-06-20 en adelante ya se miró.
    # resolve_end: velas posteriores usadas SOLO para resolver mecánicamente operaciones abiertas al cierre.
    "oos": {"scan_end": pd.Timestamp("2026-06-19 23:59:59", tz="UTC"), "months_back": 6,
            "start": pd.Timestamp("2026-01-01", tz="UTC"), "resolve_end": pd.Timestamp("2026-06-29 23:59:59", tz="UTC")},
}


def candidates(w: dict) -> tuple[pd.DataFrame, int]:
    base.MONTHS_BACK = w["months_back"]  # los workers heredan el valor (fork)
    with Pool(2) as pool:  # límite térmico (feedback_hardware_thermal_limits)
        per_symbol = pool.map(partial(ten._scan_one, end=w["scan_end"]), SYMBOLS)
    seen, unique = set(), []
    for s in (s for sigs in per_symbol for s in sigs):
        key = (s["symbol"], s["strategy"], s["setup"], s["direction"], s["sweep_timestamp"], s["mss_timestamp"], s["fvg_confirmed_at"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    sigs = pd.DataFrame(unique)
    sigs["generated_at"] = pd.to_datetime(sigs["generated_at"], utc=True)
    if w["start"] is not None:
        sigs = sigs[sigs["generated_at"] >= w["start"]]
    sigs = sigs[sigs["generated_at"] <= w["scan_end"]].reset_index(drop=True)
    info = json.loads((base.DATA_DIR / "symbol_info_full.json").read_text())
    cost = {s: estimate_symbol_cost(s, info[s]["point"], ten._load_clipped(s, "M15", w["scan_end"])) for s in SYMBOLS}
    sigs["net_rr"] = [r["risk_reward"] - cost_in_r(cost[r["symbol"]], abs(r["entry"] - r["sl"])) for _, r in sigs.iterrows()]
    n_days = (sigs["generated_at"].max() - sigs["generated_at"].min()).days or 1
    return sigs[sigs["net_rr"] >= base.MIN_NET_RR].reset_index(drop=True), len(sigs), n_days


def evaluate(cand: pd.DataFrame, w: dict, n_unique: int, n_days: int) -> dict:
    rng = np.random.default_rng(0)
    m15 = {s: ten._load_clipped(s, "M15", w["resolve_end"]) for s in SYMBOLS}
    zero = {s: 0.0 for s in SYMBOLS}
    spreads = {s: hr._spread_price(s) for s in SYMBOLS}
    scen = {"actual": None, "solo_sl_vela_entrada": hr.make_pessimistic(zero, True),
            "solo_lado_ask": hr.make_pessimistic(spreads, False), "pesimista_completo": hr.make_pessimistic(spreads, True)}
    out, frames = {"setups_unicos": n_unique, "candidatos_rr_neto_ge_2": len(cand), "dias_ventana": n_days}, {}
    for name, fn in scen.items():
        ex = hr.run(cand, m15, fn)
        frames[name] = ex
        bb = hr.block_boot(ex, lambda d: d["net_r"].mean(), rng) if len(ex) else np.array([np.nan])
        out[name] = {"n": len(ex), "ops_por_dia": round(len(ex) / n_days, 3),
                     "r_medio": round(float(ex["net_r"].mean()), 3) if len(ex) else None,
                     "total_r": round(float(ex["net_r"].sum()), 2), "ci90_bloques": hr.ci90(bb),
                     "P(r>0)": round(float(np.nanmean(bb > 0)), 3), "dias_con_ops": int(ex["day"].nunique()) if len(ex) else 0}
    return out, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=list(WINDOWS), required=True)
    w = WINDOWS[ap.parse_args().window]
    cand, n_unique, n_days = candidates(w)
    res, frames = evaluate(cand, w, n_unique, n_days)
    print(json.dumps(res, indent=2, default=str))
    label = ap.parse_args().window
    if label == "insample":
        assert len(cand) == 69 and res["actual"]["n"] == 34 and res["actual"]["r_medio"] == 0.474 \
            and res["pesimista_completo"]["r_medio"] == 0.33, "el harness NO reproduce la ventana de diseño"
        print("\nOK: el harness reproduce la ventana de diseño (69 candidatos, n=34, +0.474R, pesimista +0.330R)")
    out = hr.dl.OUT_PATH.with_name(f"oos_result_{label}_2026-10-02.json")
    trades = {k: v.drop(columns=["trace"], errors="ignore").to_dict(orient="records") for k, v in frames.items()}
    out.write_text(json.dumps({"window": {k: str(v) for k, v in w.items()}, "results": res, "trades": trades},
                              indent=2, default=str))
    print(f"guardado en {out}")


if __name__ == "__main__":
    main()
