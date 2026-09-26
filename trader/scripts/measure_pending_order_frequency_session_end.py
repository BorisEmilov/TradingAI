"""Prompt "ventana de expiración = fin de sesión" (2026-09-23), punto 5:
comparación ÚNICA y pre-registrada contra el offset fijo de 90 min ya medido
en `measure_pending_order_frequency.py` (~0.09 operaciones/día). Único
cambio: la orden límite pendiente vence al cierre de la sesión de entrada
(11:00 hora local, Londres o NY, DST-aware -- `session_risk.entry_session_end`)
en la que se generó la señal, no a `fvg_confirmed_at + 90 min`. La salida
temporal post-entrada (`exits.MAX_MINUTES_M15`) NO cambia -- no se toca acá.

Corre el mismo escaneo (`require_retracement=False`) UNA sola vez y aplica
las DOS reglas de expiración sobre el mismo conjunto de candidatos (post
R:R neto de costo), para que la comparación sea limpia -- evita re-escanear
3 meses x 3 símbolos dos veces por un cambio que solo afecta la última etapa
(conflicto E1/E2 + concurrencia + simulación de llenado)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import cost_in_r, estimate_symbol_cost
from trader.config import load_config
from trader.events import closed_candles_as_of
from trader.mtf_strategies.analysis import PrecomputedHistory
from trader.mtf_strategies.continuation import evaluate_continuation
from trader.mtf_strategies.exits import MAX_MINUTES_M15
from trader.mtf_strategies.reversal import evaluate_reversal
from trader.mtf_strategies.session_risk import (
    PositionConcurrencyState, active_entry_session, can_open_new_trade, entry_session_end,
    record_position_closed, record_position_opened,
)
from trader.mtf_strategies.signal import MTFSignal, NoSignal
from trader.mtf_strategies.trade_simulation import simulate_pending_order

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
MONTHS_BACK = 3
MIN_NET_RR = 2.0
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"
OUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_pending_order_frequency_session_end.json"


def _load(symbol: str, tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_{tf}.csv", parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df.sort_values("timestamp").reset_index(drop=True)


def _signal_to_dict(s: MTFSignal) -> dict:
    return {
        "strategy": s.strategy, "setup": s.setup, "symbol": s.symbol, "direction": s.direction,
        "session": s.session, "entry": s.entry, "sl": s.sl, "tp1": s.tp1, "tp2": s.tp2,
        "risk_reward": s.risk_reward, "sweep_timestamp": str(s.sweep_timestamp),
        "mss_timestamp": str(s.mss_timestamp), "fvg_confirmed_at": str(s.fvg_confirmed_at),
        "generated_at": str(s.generated_at), "trace": s.trace,
    }


def _to_signal(row: dict) -> MTFSignal:
    return MTFSignal(
        strategy=row["strategy"], setup=row["setup"], symbol=row["symbol"], direction=row["direction"],
        session=row["session"], entry=row["entry"], sl=row["sl"], tp1=row["tp1"], tp2=row["tp2"],
        risk_reward=row["risk_reward"], sweep_timestamp=pd.Timestamp(row["sweep_timestamp"]),
        mss_timestamp=pd.Timestamp(row["mss_timestamp"]), fvg_confirmed_at=pd.Timestamp(row["fvg_confirmed_at"]),
        generated_at=pd.Timestamp(row["generated_at"]), trace=row.get("trace", {}),
    )


def _scan(config, m15_cache: dict[str, pd.DataFrame], load: Callable[[str, str], pd.DataFrame] = _load) -> list[dict]:
    all_signals: list[dict] = []
    for symbol in m15_cache:
        t0 = time.time()
        h1_raw, m15_raw, d1_raw = load(symbol, "H1"), m15_cache[symbol], load(symbol, "D1")
        h4_raw = (
            h1_raw.set_index("timestamp").resample("4h", origin="start_day")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
        )
        window_start = m15_raw["timestamp"].iloc[-1] - pd.DateOffset(months=MONTHS_BACK)
        m15_window = m15_raw[m15_raw["timestamp"] >= window_start].reset_index(drop=True)

        h4_pre = PrecomputedHistory(h4_raw, "H4", config)
        h1_pre = PrecomputedHistory(h1_raw, "H1", config)
        m15_pre = PrecomputedHistory(m15_raw, "M15", config)

        n_evaluated = 0
        symbol_signals: list[dict] = []
        for ts in m15_window["timestamp"]:
            if active_entry_session(ts) is None:
                continue
            n_evaluated += 1
            h4, h1, m15 = h4_pre.as_of(ts), h1_pre.as_of(ts), m15_pre.as_of(ts)
            current_price = float(m15.df["close"].iloc[-1]) if len(m15.df) else float("nan")
            d1_as_of = closed_candles_as_of(d1_raw, "D1", ts)

            cont = evaluate_continuation(symbol, h4, h1, m15, d1_as_of, config, ts, current_price, require_retracement=False)
            rev = evaluate_reversal(symbol, h4, h1, m15, d1_as_of, config, ts, require_retracement=False)

            for result in (cont, rev):
                if not isinstance(result, NoSignal):
                    symbol_signals.extend(_signal_to_dict(s) for s in result)

        all_signals.extend(symbol_signals)
        print(f"{symbol}: {n_evaluated} bars en ventana de entrada, {time.time()-t0:.1f}s, "
              f"{len(symbol_signals)} señales crudas (por chequeo de vela)")
    return all_signals


def _run_conflict_concurrency_fill(
    survived: pd.DataFrame, m15_cache: dict[str, pd.DataFrame], deadline_fn: Callable[[dict], pd.Timestamp]
) -> dict:
    executed, expired_unfilled, discarded_conflict, discarded_concurrency = [], [], [], []
    for symbol in sorted(survived["symbol"].unique()):
        m15 = m15_cache[symbol]
        sym_sigs = survived[survived["symbol"] == symbol].sort_values("generated_at").reset_index(drop=True)
        state = PositionConcurrencyState()
        open_until: pd.Timestamp | None = None
        moment_counts = sym_sigs.groupby("generated_at")["strategy"].apply(lambda s: set(s))
        conflicted_moments = {t for t, strategies in moment_counts.items() if len(strategies) > 1}

        for _, row in sym_sigs.iterrows():
            row_dict = row.to_dict()
            if row["generated_at"] in conflicted_moments:
                discarded_conflict.append(row_dict)
                continue
            if open_until is not None and row["generated_at"] >= open_until:
                state = record_position_closed(state, symbol)
                open_until = None
            ok, _ = can_open_new_trade(state, symbol)
            if not ok:
                discarded_concurrency.append(row_dict)
                continue

            sig = _to_signal(row_dict)
            filled, freed_at = simulate_pending_order(sig, m15, deadline=deadline_fn(row_dict))
            row_dict["filled"] = filled
            row_dict["symbol_freed_at"] = str(freed_at) if freed_at is not None else None
            (executed if filled else expired_unfilled).append(row_dict)
            state = record_position_opened(state, symbol)
            open_until = freed_at if freed_at is not None else pd.Timestamp.max.tz_localize("UTC")

    return {
        "executed": executed, "expired_unfilled": expired_unfilled,
        "discarded_conflict": discarded_conflict, "discarded_concurrency": discarded_concurrency,
    }


def _report(label: str, result: dict, n_days: int) -> None:
    n_placed = len(result["executed"]) + len(result["expired_unfilled"])
    n_filled = len(result["executed"])
    print(f"\n=== {label} ===")
    print(f"Descartados por conflicto E1/E2: {len(result['discarded_conflict'])}")
    print(f"Descartados por concurrencia: {len(result['discarded_concurrency'])}")
    print(f"Órdenes límite colocadas: {n_placed}")
    if n_placed:
        print(f"  se LLENARON: {n_filled} ({100*n_filled/n_placed:.1f}%)")
        print(f"  EXPIRARON sin llenarse: {n_placed - n_filled} ({100*(n_placed-n_filled)/n_placed:.1f}%)")
    if result["executed"]:
        exec_df = pd.DataFrame(result["executed"])
        print("  por símbolo x estrategia:")
        print(exec_df.groupby(["symbol", "strategy"]).size().unstack(fill_value=0).to_string().replace("\n", "\n  "))
    print(f"Promedio EJECUTADAS/día: {n_filled/n_days:.3f}")


def main() -> None:
    config = load_config()
    with open(DATA_DIR / "symbol_info_full.json") as f:
        symbol_info = json.load(f)

    m15_cache = {symbol: _load(symbol, "M15") for symbol in SYMBOLS}
    all_signals = _scan(config, m15_cache)

    dedup_keys = set()
    unique = []
    for s in all_signals:
        key = (s["symbol"], s["strategy"], s["setup"], s["direction"], s["sweep_timestamp"], s["mss_timestamp"], s["fvg_confirmed_at"])
        if key not in dedup_keys:
            dedup_keys.add(key)
            unique.append(s)
    sigs = pd.DataFrame(unique)
    sigs["generated_at"] = pd.to_datetime(sigs["generated_at"])
    sigs["fvg_confirmed_at"] = pd.to_datetime(sigs["fvg_confirmed_at"])
    print(f"\n{len(all_signals)} señales crudas -> {len(sigs)} setups únicos tras deduplicar")

    cost_by_symbol = {s: estimate_symbol_cost(s, symbol_info[s]["point"], m15_cache[s]) for s in SYMBOLS}
    net_rr = []
    for _, row in sigs.iterrows():
        risk_price = abs(row["entry"] - row["sl"])
        net_rr.append(row["risk_reward"] - cost_in_r(cost_by_symbol[row["symbol"]], risk_price))
    sigs["net_rr"] = net_rr
    survived = sigs[sigs["net_rr"] >= MIN_NET_RR].reset_index(drop=True)
    print(f"Candidatos con R:R NETO de costo >= {MIN_NET_RR}: {len(survived)} "
          f"(descartados solo por costo: {len(sigs) - len(survived)})")

    n_days = (sigs["generated_at"].max() - sigs["generated_at"].min()).days or 1

    fixed_90min = _run_conflict_concurrency_fill(
        survived, m15_cache, lambda row: pd.Timestamp(row["fvg_confirmed_at"]) + pd.Timedelta(minutes=MAX_MINUTES_M15)
    )
    session_end = _run_conflict_concurrency_fill(
        survived, m15_cache, lambda row: entry_session_end(row["session"], pd.Timestamp(row["generated_at"]))
    )

    _report(f"MODELO ANTERIOR: expiración = fvg_confirmed_at + {MAX_MINUTES_M15} min", fixed_90min, n_days)
    _report("MODELO NUEVO: expiración = fin de sesión de entrada (11:00 local, DST-aware)", session_end, n_days)

    n_filled_90 = len(fixed_90min["executed"])
    n_filled_session = len(session_end["executed"])
    print(f"\n=== COMPARACIÓN DIRECTA (mismos {len(survived)} candidatos, misma metodología) ===")
    print(f"Offset fijo 90 min:  {n_filled_90} ejecutadas -> {n_filled_90/n_days:.3f}/día")
    print(f"Fin de sesión:       {n_filled_session} ejecutadas -> {n_filled_session/n_days:.3f}/día")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "min_net_rr": MIN_NET_RR, "n_candidates_net_rr_ge_2": len(survived), "n_days": n_days,
            "fixed_90min": {**fixed_90min, "n_filled": n_filled_90, "executed_per_day": n_filled_90 / n_days},
            "session_end": {**session_end, "n_filled": n_filled_session, "executed_per_day": n_filled_session / n_days},
        }, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
