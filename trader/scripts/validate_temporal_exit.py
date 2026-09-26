"""Prompt "validar la salida temporal (90 min)" (2026-09-25): comparación
ÚNICA con vs. sin salida temporal sobre las operaciones ya generadas en la
medición de 3 meses / 10 símbolos (orden límite llenada, con corte diario).

Mismo motor (`trade_simulation`), solo `temporal_exit=False` -- mismo SL/TP1/
breakeven/TP2/geometría. R neto del mismo costo de spread del gate.

A) MISMAS operaciones (lo pedido): cada una se simula con y sin la regla.
   Si la regla no se dispara ambos caminos son idénticos, así que las que
   difieren son exactamente las que la regla cortó.
B) Robustez: pipeline completo re-corrido sin la regla (sin salida temporal
   las posiciones duran más -> la concurrencia por símbolo cambia qué
   operaciones existen).
No se prueba ningún otro umbral. No toca el piloto en vivo.
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
import measure_pending_order_frequency_session_end as base
from trader.mtf_strategies.session_risk import entry_session_end
from trader.mtf_strategies.trade_simulation import simulate_pending_order_outcome

OUT_PATH = dl.OUT_PATH.with_name("temporal_exit_validation.json")
BOOTSTRAP_N = 10_000


def _max_drawdown(r: pd.Series) -> float:
    equity = r.cumsum()
    return float((equity.cummax().clip(lower=0) - equity).max()) if len(r) else 0.0


def _stats(df: pd.DataFrame, r_col: str, ts_col: str) -> dict:
    res = df.dropna(subset=[r_col]).sort_values(ts_col)
    r = res[r_col]
    return {"n": int(len(df)), "n_resolved": int(len(r)), "expectancy_r": round(r.mean(), 3),
            "total_r": round(r.sum(), 2), "win_rate": round(float((r > 0).mean()), 3),
            "max_drawdown_r": round(_max_drawdown(r.reset_index(drop=True)), 2)}


def _destination(r_gross: float | None, rr: float) -> str:
    if r_gross is None or pd.isna(r_gross):
        return "sin resolver (datos/límite de búsqueda)"
    if abs(r_gross + 1.0) < 1e-6:
        return "SL completo (-1R)"
    if abs(r_gross - 0.5) < 1e-6:
        return "TP1 + breakeven (+0.5R)"
    if abs(r_gross - (0.5 + 0.5 * rr)) < 1e-6:
        return "TP1 + TP2"
    return f"otro ({r_gross:+.2f}R)"


def main() -> None:
    with open(dl.OUT_PATH) as f:
        executed = json.load(f)["with_daily_loss"]["executed"]
    with open(ten.OUT_PATH) as f:
        prev = json.load(f)
    end = pd.Timestamp(prev["common_end"])
    m15 = {s: ten._load_clipped(s, "M15", end) for s in prev["symbols"]}

    # -- A) mismas operaciones ------------------------------------------------
    rows = []
    for row in executed:
        sig = base._to_signal(row)
        deadline = entry_session_end(row["session"], pd.Timestamp(row["generated_at"]))
        cost_r = row["risk_reward"] - row["net_rr"]
        _, close_w, entry_t, r_w = simulate_pending_order_outcome(sig, m15[row["symbol"]], deadline, temporal_exit=True)
        _, close_wo, _, r_wo = simulate_pending_order_outcome(sig, m15[row["symbol"]], deadline, temporal_exit=False)
        rows.append({
            "symbol": row["symbol"], "strategy": row["strategy"], "setup": row["setup"], "entry_time": entry_t,
            "rr": sig.risk_reward, "close_with": close_w, "close_without": close_wo,
            "r_with_gross": r_w, "r_without_gross": r_wo,
            "r_with": None if r_w is None else r_w - cost_r, "r_without": None if r_wo is None else r_wo - cost_r,
            "cut_by_rule": (close_w, r_w) != (close_wo, r_wo),
        })
    df = pd.DataFrame(rows)
    assert (df.loc[~df["cut_by_rule"], "r_with"].fillna(-99) == df.loc[~df["cut_by_rule"], "r_without"].fillna(-99)).all()

    with_stats, without_stats = _stats(df, "r_with", "close_with"), _stats(df, "r_without", "close_without")
    cut = df[df["cut_by_rule"]].copy()
    cut["destino_sin_regla"] = [_destination(r, rr) for r, rr in zip(cut["r_without_gross"], cut["rr"])]

    print(f"=== A) MISMAS {len(df)} operaciones, con vs. sin salida temporal (R neto de spread) ===")
    print(pd.DataFrame({"con_regla": with_stats, "sin_regla": without_stats}).to_string())
    print(f"\nla regla cortó {len(cut)} de {len(df)} operaciones. Destino si hubieran seguido corriendo:")
    print(cut["destino_sin_regla"].value_counts().to_string())
    print(f"\nR de esas {len(cut)}: con regla suma={cut['r_with'].sum():+.2f}R (media {cut['r_with'].mean():+.3f}) | "
          f"sin regla suma={cut['r_without'].sum():+.2f}R (media {cut['r_without'].mean():+.3f})")
    print("\ndetalle de las cortadas:")
    print(cut[["symbol", "strategy", "setup", "entry_time", "r_with", "r_without", "destino_sin_regla"]]
          .round(3).to_string(index=False))

    # bootstrap pareado de la diferencia de expectancy (solo operaciones resueltas en ambos)
    both = df.dropna(subset=["r_with", "r_without"])
    diff = (both["r_without"] - both["r_with"]).to_numpy()
    rng = np.random.default_rng(0)
    boots = rng.choice(diff, size=(BOOTSTRAP_N, len(diff)), replace=True).mean(axis=1)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"\ndiferencia de expectancy (sin - con) = {diff.mean():+.3f}R por operación, IC95% bootstrap "
          f"[{ci[0]:+.3f}, {ci[1]:+.3f}] (n={len(diff)}), P(sin regla mejor)={float((boots > 0).mean()):.2f}")

    # -- B) pipeline completo sin la regla -------------------------------------
    cand_rows = [r for k in ("executed", "expired_unfilled", "discarded_conflict", "discarded_concurrency") for r in prev[k]]
    survived = pd.DataFrame(cand_rows).drop(columns=["filled", "symbol_freed_at"], errors="ignore")
    survived["generated_at"] = pd.to_datetime(survived["generated_at"], utc=True)
    pipe = {}
    for label, te in (("con_regla", True), ("sin_regla", False)):
        res = dl._simulate(survived, m15, apply_daily_loss=True, temporal_exit=te)
        ex = pd.DataFrame(res["executed"])
        ex["generated_at"] = pd.to_datetime(ex["generated_at"], utc=True)
        pipe[label] = {**_stats(ex, "net_r", "generated_at"), "concurrencia_descartadas": len(res["discarded_concurrency"]),
                       "cortes_diarios": len(res["lockouts"])}
    print("\n=== B) robustez: pipeline completo re-corrido (concurrencia + corte diario incluidos) ===")
    print(pd.DataFrame(pipe).to_string())

    with open(OUT_PATH, "w") as f:
        json.dump({"same_trades": {"con_regla": with_stats, "sin_regla": without_stats},
                   "cut_trades": cut.to_dict(orient="records"),
                   "expectancy_diff": {"mean": float(diff.mean()), "ci95": ci, "n": int(len(diff))},
                   "full_pipeline": pipe}, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
