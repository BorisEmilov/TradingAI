"""Prompt "chequeo de precio pre-envío + alinear el simulador" (2026-09-29), punto 5:
frecuencia re-medida con el simulador alineado al piloto (orden colocada en
`generated_at`, solo si el precio no cruzó la entrada). Mismos 66 candidatos
(R:R neto >= 2) reconstruidos de logs/mtf_frequency_10_symbols.json, mismo
orden de filtros y corte diario -1.5R que `measure_frequency_10_symbols_daily_loss`.
Referencia previa (simulador viejo): 41 ejecutadas con salida temporal, 40 sin ella."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import measure_frequency_10_symbols as ten
import measure_frequency_10_symbols_daily_loss as dl
import measure_pending_order_frequency_session_end as base
from trader.risk.levels import limit_entry_still_ahead

OUT_PATH = dl.OUT_PATH.with_name("mtf_frequency_after_placement_check.json")
PREVIOUS = {True: 41, False: 40}


def main() -> None:
    prev = json.loads(ten.OUT_PATH.read_text())
    rows = [r for k in ("executed", "expired_unfilled", "discarded_conflict", "discarded_concurrency") for r in prev[k]]
    survived = pd.DataFrame(rows).drop(columns=["filled", "symbol_freed_at"], errors="ignore")
    survived["generated_at"] = pd.to_datetime(survived["generated_at"], utc=True)
    n_days, end = prev["n_days"], pd.Timestamp(prev["common_end"])
    m15 = {s: ten._load_clipped(s, "M15", end) for s in prev["symbols"]}

    out = {}
    for te in (False, True):
        res = dl._simulate(survived, m15, apply_daily_loss=True, temporal_exit=te)
        not_placeable = []
        for r in res["expired_unfilled"]:
            df, sig = m15[r["symbol"]], base._to_signal(r)
            placed = df[df["timestamp"] >= sig.generated_at]
            if len(placed) and not limit_entry_still_ahead(sig.direction, float(placed["open"].iloc[0]), sig.entry, sig.tp2):
                not_placeable.append(r)
        r_net = pd.Series([r["net_r"] for r in res["executed"] if r["net_r"] is not None])
        eq = r_net.cumsum()
        n = len(res["executed"])
        label = "sin_salida_temporal (piloto actual)" if not te else "con_salida_temporal"
        out[label] = {
            "ejecutadas_antes": PREVIOUS[te], "ejecutadas_ahora": n, "por_dia": round(n / n_days, 3),
            "por_semana": round(n / n_days * 7, 2), "descartadas_precio_pasado_entrada": len(not_placeable),
            "expiradas_sin_llenar": len(res["expired_unfilled"]) - len(not_placeable),
            "concurrencia": len(res["discarded_concurrency"]), "cortes_diarios": len(res["lockouts"]),
            "expectancy_r": round(r_net.mean(), 3), "total_r": round(r_net.sum(), 2),
            "max_drawdown_r": round(float((eq.cummax().clip(lower=0) - eq).max()), 2),
        }
        out[label]["_executed"] = res["executed"]
    table = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in out.items()})
    print(f"ventana {n_days} días, 10 símbolos, corte diario -1.5R\n{table.to_string()}")
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
