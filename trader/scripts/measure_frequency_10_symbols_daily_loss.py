"""Prompt "2 verificaciones antes de relanzar con 10 símbolos" (2026-09-25),
punto 1: re-medir la frecuencia de 10 símbolos CON el corte diario global
-1.5R, reusando `DailyLossState`/`record_daily_result`/
`daily_loss_limit_reached` del piloto (bloquea entradas nuevas, nunca cierra
posiciones abiertas).

No re-escanea: los 66 candidatos con R:R neto >= 2 se reconstruyen desde
`logs/mtf_frequency_10_symbols.json` (cada uno cayó en exactamente una de
executed/expired_unfilled/discarded_conflict/discarded_concurrency).

A diferencia de la medición por símbolo, el corte es GLOBAL, así que acá se
simula en UN solo orden cronológico con todos los símbolos, igual que el
piloto en vivo:
- orden de filtros = piloto: conflicto E1/E2 -> concurrencia -> corte diario;
- el R de cada operación se registra en el día de su CIERRE (como
  `_finalize_close`), neto del mismo costo de spread usado en el gate;
- al activarse el corte se cancelan las órdenes límite todavía sin llenar
  (como `_manage_pending_orders`); las posiciones ya llenadas siguen su curso.
Sanity check: con el corte desactivado debe reproducir las 34 ejecutadas.
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import measure_pending_order_frequency_session_end as base
import measure_frequency_10_symbols as ten
from trader.mtf_strategies.session_risk import (
    DailyLossState, daily_loss_limit_reached, entry_session_end, record_daily_result,
)
from trader.mtf_strategies.trade_simulation import simulate_pending_order_outcome

IN_PATH = ten.OUT_PATH
OUT_PATH = IN_PATH.with_name("mtf_frequency_10_symbols_daily_loss.json")
FAR_FUTURE = pd.Timestamp.max.tz_localize("UTC")


def _simulate(survived: pd.DataFrame, m15_cache: dict[str, pd.DataFrame], apply_daily_loss: bool,
              temporal_exit: bool = True) -> dict:
    sigs = survived.sort_values(["generated_at", "symbol"], kind="stable").reset_index(drop=True)
    conflicted = {
        (sym, t) for (sym, t), strategies in sigs.groupby(["symbol", "generated_at"])["strategy"].apply(set).items()
        if len(strategies) > 1
    }
    out = {k: [] for k in ("executed", "expired_unfilled", "discarded_conflict", "discarded_concurrency",
                           "discarded_daily_loss", "cancelled_daily_loss")}
    open_until: dict[str, pd.Timestamp] = {}
    placed: list[dict] = []  # órdenes colocadas (para cancelar las pendientes si salta el corte)
    closes: list[tuple] = []  # heap de (cierre, idx) de posiciones llenadas
    daily = DailyLossState(date=sigs["generated_at"].iloc[0].normalize())
    lockouts: list[dict] = []

    def apply_closes_until(t: pd.Timestamp) -> None:
        nonlocal daily
        while closes and closes[0][0] <= t:
            close_ts, idx = heapq.heappop(closes)
            trade = placed[idx]
            if trade["cancelled"]:
                continue
            was_locked = daily.locked_out and daily.date == close_ts.normalize()
            daily = record_daily_result(daily, close_ts, trade["net_r"])
            if daily.locked_out and not was_locked:
                lockouts.append({"at": str(close_ts), "cumulative_r": round(daily.cumulative_r, 3)})
                for other in placed:  # cancelar órdenes límite aún sin llenar
                    pending = not other["cancelled"] and other["placed_at"] <= close_ts < other["pending_until"]
                    if pending:
                        other["cancelled"] = True
                        open_until[other["symbol"]] = close_ts
                        if other["filled"]:
                            out["cancelled_daily_loss"].append(other["row"])

    for _, row in sigs.iterrows():
        t, symbol, row_dict = row["generated_at"], row["symbol"], row.to_dict()
        if apply_daily_loss:
            apply_closes_until(t)
        if (symbol, t) in conflicted:
            out["discarded_conflict"].append(row_dict)
            continue
        if symbol in open_until and t >= open_until[symbol]:
            del open_until[symbol]
        if symbol in open_until:
            out["discarded_concurrency"].append(row_dict)
            continue
        if apply_daily_loss:
            daily, locked = daily_loss_limit_reached(daily, t)
            if locked:
                out["discarded_daily_loss"].append(row_dict)
                continue

        deadline = entry_session_end(row["session"], pd.Timestamp(t))
        filled, freed_at, entry_time, r = simulate_pending_order_outcome(
            base._to_signal(row_dict), m15_cache[symbol], deadline, temporal_exit=temporal_exit)
        cost_r = row["risk_reward"] - row["net_rr"]
        placed.append({
            "symbol": symbol, "row": row_dict, "filled": filled, "cancelled": False, "placed_at": t,
            # pendiente (cancelable) hasta que se llena, o hasta que vence si nunca se llena
            "pending_until": entry_time if filled else deadline,
            "net_r": None if r is None else r - cost_r,
        })
        open_until[symbol] = freed_at if freed_at is not None else FAR_FUTURE
        if filled and freed_at is not None and r is not None:
            heapq.heappush(closes, (freed_at, len(placed) - 1))

    if apply_daily_loss:
        apply_closes_until(FAR_FUTURE)  # cancelaciones por cortes posteriores a la última señal

    for trade in placed:
        if trade["cancelled"]:
            continue
        row_dict = {**trade["row"], "net_r": trade["net_r"]}
        (out["executed"] if trade["filled"] else out["expired_unfilled"]).append(row_dict)
    out["lockouts"] = lockouts
    return out


def main() -> None:
    with open(IN_PATH) as f:
        prev = json.load(f)
    rows = [r for k in ("executed", "expired_unfilled", "discarded_conflict", "discarded_concurrency") for r in prev[k]]
    survived = pd.DataFrame(rows).drop(columns=["filled", "symbol_freed_at"], errors="ignore")
    survived["generated_at"] = pd.to_datetime(survived["generated_at"], utc=True)
    symbols, n_days, end = prev["symbols"], prev["n_days"], pd.Timestamp(prev["common_end"])
    print(f"{len(survived)} candidatos reconstruidos, {len(symbols)} símbolos, ventana {n_days} días")

    m15_cache = {s: ten._load_clipped(s, "M15", end) for s in symbols}
    without = _simulate(survived, m15_cache, apply_daily_loss=False)
    with_dl = _simulate(survived, m15_cache, apply_daily_loss=True)
    assert len(without["executed"]) == len(prev["executed"]), "no reproduce la medición previa"

    def per_symbol(res: dict, key: str) -> pd.Series:
        return pd.Series([r["symbol"] for r in res[key]], dtype=object).value_counts()

    table = pd.DataFrame({
        "ejec_sin_corte": per_symbol(without, "executed"),
        "ejec_con_corte": per_symbol(with_dl, "executed"),
        "bloq_corte_señal": per_symbol(with_dl, "discarded_daily_loss"),
        "cancel_corte_pend": per_symbol(with_dl, "cancelled_daily_loss"),
    }).reindex(symbols).fillna(0).astype(int)
    table.loc["TOTAL"] = table.sum()
    table["por_dia"] = (table["ejec_con_corte"] / n_days).round(3)
    table["por_semana"] = (table["ejec_con_corte"] / n_days * 7).round(2)
    print(table.to_string())

    n_with, n_without = len(with_dl["executed"]), len(without["executed"])
    net_r = pd.Series([r["net_r"] for r in with_dl["executed"] if r["net_r"] is not None])
    print(f"\ncortes diarios disparados: {len(with_dl['lockouts'])} -> {with_dl['lockouts']}")
    print(f"señales bloqueadas por corte (ni se colocaron): {len(with_dl['discarded_daily_loss'])} "
          f"(de ellas, cuántas se hubieran llenado no se sabe sin simular -- se reportan como bloqueadas)")
    print(f"órdenes pendientes canceladas por corte que SÍ se habrían llenado: {len(with_dl['cancelled_daily_loss'])}")
    print(f"concurrencia/expiradas cambian también (símbolos liberados antes): "
          f"concurrencia {len(without['discarded_concurrency'])}->{len(with_dl['discarded_concurrency'])}, "
          f"expiradas {len(without['expired_unfilled'])}->{len(with_dl['expired_unfilled'])}")
    print(f"\nEJECUTADAS: sin corte {n_without} ({n_without/n_days:.3f}/día) -> con corte {n_with} "
          f"({n_with/n_days:.3f}/día, {n_with/n_days*7:.2f}/semana); pérdida neta {n_without - n_with}")
    print(f"(contexto, no es métrica de edge: R neto simulado de las ejecutadas -- n={len(net_r)}, "
          f"suma={net_r.sum():+.2f}R, media={net_r.mean():+.3f}R)")

    with open(OUT_PATH, "w") as f:
        json.dump({"n_days": n_days, "without_daily_loss": without, "with_daily_loss": with_dl,
                   "table": table.reset_index().to_dict(orient="records")}, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
