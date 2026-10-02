"""Prompt "evaluación más honesta" (2026-10-02). Solo re-evaluación estadística:
ninguna regla ni parámetro de la estrategia cambia, y el motor
(`trade_simulation.py`) no se toca -- el escenario pesimista es una función
aparte que se inyecta en el mismo pipeline (`dl._simulate`: conflicto,
concurrencia, corte diario -1.5R, parcial+BE, expiración fin de sesión).

1) IC 90% del R medio por operación con bootstrap por BLOQUES DIARIOS (se
   remuestrean días de generación de la señal, con todas sus operaciones).
2) Simulador pesimista:
   a) SL dentro de la vela de entrada: si el rango de la vela del fill llega
      al SL, se cuenta -1R (el motor actual ignora esa vela).
   b) Lado ask: las velas son bid. Un BUY_LIMIT solo se llena si el ask
      (bid + spread) toca la entrada; las salidas de un corto (SL/TP, que son
      compras) se disparan por el ask. Spread = el MAYOR promedio de ticks
      medido por símbolo (logs/spread_source_verification.json: may-2026 y
      sep-2026), no el campo de la vela (que es el mínimo, ~0).
   Self-check: con spread 0 y sin (a) reproduce exactamente el backtest actual.
3) Robustez: R:R neto >= 3 vs resto, Londres vs NY -- diferencia de medias con
   el mismo bootstrap por bloques.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import measure_frequency_10_symbols as ten
import measure_frequency_10_symbols_daily_loss as dl
from trader.events import TF_DURATION
from trader.risk.levels import limit_entry_still_ahead

OUT_PATH = dl.OUT_PATH.with_name("honest_reevaluation.json")
BOOT_N = 10_000
STEP = TF_DURATION["M15"]
_engine_outcome = dl.simulate_pending_order_outcome


def _spread_price(symbol: str) -> float:
    windows = json.loads((dl.OUT_PATH.parent / "spread_source_verification.json").read_text())["windows"]
    pts = max(w[symbol]["tick_spread_mean"] for w in windows.values())
    return pts * (0.001 if symbol.endswith("JPY") else 0.00001)


def make_pessimistic(spread_by_symbol: dict[str, float], sl_in_entry_bar: bool):
    def outcome(sig, m15_df, deadline, temporal_exit=False):
        assert not temporal_exit
        sp = spread_by_symbol[sig.symbol]
        long_ = sig.direction == "long"
        ask_adj = 0.0 if long_ else sp  # salidas del corto se ejecutan al ask
        ts = m15_df["timestamp"]
        close_ts = ts + STEP
        placed = m15_df[ts >= sig.generated_at]
        if len(placed) == 0:
            return False, sig.generated_at, None, None
        o = float(placed["open"].iloc[0])
        if not limit_entry_still_ahead(sig.direction, o + sp if long_ else o, sig.entry, sig.tp2):
            return False, sig.generated_at, None, None
        window = m15_df[(ts >= sig.generated_at) & (close_ts <= deadline)]
        touch = (window["low"] + sp <= sig.entry) if long_ else (window["high"] >= sig.entry)
        if not touch.any():
            return False, deadline, None, None
        i = m15_df.index.get_loc(touch[touch].index[0])
        risk = abs(sig.entry - sig.sl)
        r_at = lambda p: ((p - sig.entry) if long_ else (sig.entry - p)) / risk
        sl_hit = lambda row, sl: (row["low"] <= sl) if long_ else (row["high"] + ask_adj >= sl)
        tp_hit = lambda row, tp: (row["high"] >= tp) if long_ else (row["low"] + ask_adj <= tp)

        if sl_in_entry_bar and sl_hit(m15_df.iloc[i], sig.sl):
            return True, close_ts.iloc[i], close_ts.iloc[i], -1.0
        partial, sl = False, sig.sl
        for j in range(i + 1, min(i + 700, len(m15_df) - 1) + 1):  # mismo límite de búsqueda que el motor
            row = m15_df.iloc[j]
            if sl_hit(row, sl):
                return True, close_ts.iloc[j], close_ts.iloc[i], (0.5 * r_at(sig.tp1) if partial else -1.0)
            if not partial:
                if tp_hit(row, sig.tp1):
                    partial, sl = True, sig.entry
            elif tp_hit(row, sig.tp2):
                return True, close_ts.iloc[j], close_ts.iloc[i], 0.5 * r_at(sig.tp1) + 0.5 * r_at(sig.tp2)
        return True, None, close_ts.iloc[i], None
    return outcome


def run(cand, m15, outcome=None) -> pd.DataFrame:
    dl.simulate_pending_order_outcome = outcome or _engine_outcome
    try:
        res = dl._simulate(cand, m15, apply_daily_loss=True, temporal_exit=False)
    finally:
        dl.simulate_pending_order_outcome = _engine_outcome
    ex = pd.DataFrame(res["executed"]).dropna(subset=["net_r"])
    ex["day"] = pd.to_datetime(ex["generated_at"], utc=True).dt.normalize()
    return ex.reset_index(drop=True)


def block_boot(ex: pd.DataFrame, stat, rng) -> np.ndarray:
    """Remuestrea DÍAS con reemplazo (cada día arrastra todas sus operaciones)."""
    days = ex["day"].unique()
    groups = [ex.index[ex["day"] == d].to_numpy() for d in days]
    out = np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        out[b] = stat(ex.loc[idx])
    return out


def ci90(x) -> list[float]:
    x = x[~np.isnan(x)]
    return [round(float(np.percentile(x, 5)), 3), round(float(np.percentile(x, 95)), 3)]


def subset_diff(ex, mask_fn, rng) -> dict:
    def diff(d):
        m = mask_fn(d)
        return d.loc[m, "net_r"].mean() - d.loc[~m, "net_r"].mean() if m.any() and (~m).any() else np.nan
    m = mask_fn(ex)
    b_in = block_boot(ex[m].reset_index(drop=True), lambda d: d["net_r"].mean(), rng)
    b_out = block_boot(ex[~m].reset_index(drop=True), lambda d: d["net_r"].mean(), rng)
    bd = block_boot(ex, diff, rng)
    return {"n_subset": int(m.sum()), "r_subset": round(ex.loc[m, "net_r"].mean(), 3), "ci90_subset": ci90(b_in),
            "n_resto": int((~m).sum()), "r_resto": round(ex.loc[~m, "net_r"].mean(), 3), "ci90_resto": ci90(b_out),
            "share_total_r": round(ex.loc[m, "net_r"].sum() / ex["net_r"].sum(), 2),
            "diff": round(float(diff(ex)), 3), "ci90_diff": ci90(bd), "P(diff>0)": round(float(np.nanmean(bd > 0)), 3)}


def main() -> None:
    rng = np.random.default_rng(0)
    prev = json.loads(ten.OUT_PATH.read_text())
    cand = pd.DataFrame([r for k in ("executed", "expired_unfilled", "discarded_conflict", "discarded_concurrency")
                         for r in prev[k]]).drop(columns=["filled", "symbol_freed_at"], errors="ignore")
    cand["generated_at"] = pd.to_datetime(cand["generated_at"], utc=True)
    m15 = {s: ten._load_clipped(s, "M15", pd.Timestamp(prev["common_end"])) for s in prev["symbols"]}

    base = run(cand, m15)
    zero = {s: 0.0 for s in prev["symbols"]}
    check = run(cand, m15, make_pessimistic(zero, sl_in_entry_bar=False))
    assert np.allclose(sorted(base["net_r"]), sorted(check["net_r"])) and len(base) == len(check), "self-check falló"

    # 1) IC por bloques diarios
    mean = lambda d: d["net_r"].mean()
    b = block_boot(base, mean, rng)
    per_day = base.groupby("day").size()
    real = {"n": len(base), "dias": int(per_day.size), "max_ops_mismo_dia": int(per_day.max()),
            "r_medio": round(base["net_r"].mean(), 3), "ci90_bloques": ci90(b), "P(r>0)": round(float((b > 0).mean()), 3)}
    print(f"=== 1) backtest actual: {real}")

    # 2) pesimista
    spreads = {s: _spread_price(s) for s in prev["symbols"]}
    scen = {"solo_sl_vela_entrada": make_pessimistic(zero, True), "solo_lado_ask": make_pessimistic(spreads, False),
            "pesimista_completo": make_pessimistic(spreads, True)}
    pess = {}
    for name, fn in scen.items():
        ex = run(cand, m15, fn)
        bb = block_boot(ex, mean, rng)
        pess[name] = {"n": len(ex), "r_medio": round(ex["net_r"].mean(), 3), "total_r": round(ex["net_r"].sum(), 2),
                      "ci90_bloques": ci90(bb), "P(r>0)": round(float((bb > 0).mean()), 3)}
        if name == "pesimista_completo":
            pess_ex = ex
    print("\n=== 2) escenarios pesimistas ===\n" + pd.DataFrame(pess).T.to_string())

    # 3) robustez (sobre el backtest actual y sobre el pesimista)
    rob = {}
    for label, ex in (("actual", base), ("pesimista", pess_ex)):
        rob[label] = {"rr_neto>=3": subset_diff(ex, lambda d: d["net_rr"] >= 3, rng),
                      "londres": subset_diff(ex, lambda d: d["session"] == "london", rng)}
        print(f"\n=== 3) robustez ({label}) ===\n" + pd.DataFrame(rob[label]).T.to_string())

    OUT_PATH.write_text(json.dumps({"real": real, "pesimista": pess, "spread_pts_usado": spreads, "robustez": rob},
                                   indent=2, default=str))
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
