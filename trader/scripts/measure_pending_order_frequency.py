"""Prompt (migración a orden límite real), punto 6: re-mide la frecuencia
real bajo el modelo nuevo -- "¿el precio tocó el nivel del 50% FVG dentro de
la ventana de 90 min?", no "¿estaba ahí en el momento exacto en que el bot
evaluó?" (esa era la pregunta de `measure_rr_prevalidation_impact.py`, que
ya no aplica: bajo el modelo nuevo no se manda una orden a mercado, se
coloca una orden límite real que espera).

Reproduce la MISMA cadena de 3 etapas que ya se usó para el número original
(~2.47/día, ver `scripts/scan_mtf_strategies.py` ->
`scripts/recompute_frequency_with_costs.py`), en un solo script:
  1. Escaneo histórico con `require_retracement=False` (el setup se detecta
     apenas MSS+FVG están confirmados, no cuando el precio ya volvió).
  2. R:R neto de costo real (spread) >= MIN_NET_RR.
  3. Conflicto E1/E2 (mismo símbolo, mismo momento de detección) +
     concurrencia por símbolo -- reusa `simulate_pending_order` (orden límite
     con expiración a 90 min, ver trade_simulation.py) para saber si cada
     candidato se llena y cuándo se libera el símbolo.

Momento de decisión para conflicto/concurrencia: `generated_at` (igual que
el método original) -- bajo el modelo nuevo, ya no es "cuándo se vio el
retroceso" sino prácticamente "cuándo se confirmó MSS+FVG" (el gate ya no
espera nada más), así que sigue siendo el momento correcto: la primera vez
que el setup existe como candidato ejecutable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
    PositionConcurrencyState, active_entry_session, can_open_new_trade, record_position_closed, record_position_opened,
)
from trader.mtf_strategies.signal import MTFSignal, NoSignal
from trader.mtf_strategies.trade_simulation import simulate_pending_order

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
MONTHS_BACK = 3
MIN_NET_RR = 2.0
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"
OUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_pending_order_frequency.json"


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


def _scan(config, m15_cache: dict[str, pd.DataFrame]) -> list[dict]:
    all_signals: list[dict] = []
    for symbol in SYMBOLS:
        t0 = time.time()
        h1_raw, m15_raw, d1_raw = _load(symbol, "H1"), m15_cache[symbol], _load(symbol, "D1")
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


def main() -> None:
    config = load_config()
    with open(DATA_DIR / "symbol_info_full.json") as f:
        symbol_info = json.load(f)

    m15_cache = {symbol: _load(symbol, "M15") for symbol in SYMBOLS}

    all_signals = _scan(config, m15_cache)

    # dedup por identidad real del setup -- mismo criterio que scan_mtf_strategies.py
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
    print(f"\n{len(all_signals)} señales crudas -> {len(sigs)} setups únicos tras deduplicar "
          "(candidatos: MSS+FVG confirmados, R:R teórico>=2, SIN esperar retroceso)")

    # -- R:R neto de costo real --
    cost_by_symbol = {s: estimate_symbol_cost(s, symbol_info[s]["point"], m15_cache[s]) for s in SYMBOLS}
    net_rr = []
    for _, row in sigs.iterrows():
        risk_price = abs(row["entry"] - row["sl"])
        net_rr.append(row["risk_reward"] - cost_in_r(cost_by_symbol[row["symbol"]], risk_price))
    sigs["net_rr"] = net_rr
    survived = sigs[sigs["net_rr"] >= MIN_NET_RR].reset_index(drop=True)
    print(f"Candidatos con R:R NETO de costo >= {MIN_NET_RR}: {len(survived)} "
          f"(descartados solo por costo: {len(sigs) - len(survived)})")

    # -- conflicto E1/E2 + concurrencia por símbolo, con orden límite real (expira a 90 min) --
    executed, expired_unfilled, discarded_conflict, discarded_concurrency = [], [], [], []
    for symbol in sorted(survived["symbol"].unique()):
        m15 = m15_cache[symbol]
        sym_sigs = survived[survived["symbol"] == symbol].sort_values("generated_at").reset_index(drop=True)
        state = PositionConcurrencyState()
        open_until: pd.Timestamp | None = None
        moment_counts = sym_sigs.groupby("generated_at")["strategy"].apply(lambda s: set(s))
        conflicted_moments = {t for t, strategies in moment_counts.items() if len(strategies) > 1}

        for _, row in sym_sigs.iterrows():
            if row["generated_at"] in conflicted_moments:
                discarded_conflict.append(row.to_dict())
                continue
            if open_until is not None and row["generated_at"] >= open_until:
                state = record_position_closed(state, symbol)
                open_until = None
            ok, _ = can_open_new_trade(state, symbol)
            if not ok:
                discarded_concurrency.append(row.to_dict())
                continue

            sig = _to_signal(row.to_dict())
            filled, freed_at = simulate_pending_order(sig, m15, max_wait_minutes=MAX_MINUTES_M15)
            row_dict = row.to_dict()
            row_dict["filled"] = filled
            row_dict["symbol_freed_at"] = str(freed_at) if freed_at is not None else None
            (executed if filled else expired_unfilled).append(row_dict)
            state = record_position_opened(state, symbol)
            open_until = freed_at if freed_at is not None else pd.Timestamp.max.tz_localize("UTC")

    exec_df = pd.DataFrame(executed)
    n_placed = len(executed) + len(expired_unfilled)
    n_days = (sigs["generated_at"].max() - sigs["generated_at"].min()).days or 1

    print(f"\n=== RESULTADO: modelo de orden límite real (expira a {MAX_MINUTES_M15} min) ===")
    print(f"Descartados por conflicto E1/E2: {len(discarded_conflict)}")
    print(f"Descartados por concurrencia (símbolo ya ocupado -- pendiente o posición abierta): {len(discarded_concurrency)}")
    print(f"Órdenes límite colocadas (sobreviven conflicto+concurrencia): {n_placed}")
    print(f"  se LLENARON (precio tocó el nivel dentro de {MAX_MINUTES_M15} min): {len(executed)} "
          f"({100*len(executed)/n_placed:.1f}%)" if n_placed else "")
    print(f"  EXPIRARON sin llenarse: {len(expired_unfilled)} "
          f"({100*len(expired_unfilled)/n_placed:.1f}%)" if n_placed else "")
    if len(exec_df):
        print("\nOperaciones ejecutadas (llenadas), por símbolo x estrategia:")
        print(exec_df.groupby(["symbol", "strategy"]).size().unstack(fill_value=0))
    print(f"\nVentana ~{n_days} días")
    print(f"Promedio EJECUTADAS/día (los 3 símbolos): {len(executed)/n_days:.2f}")
    print(f"  (para comparar: ~2.47/día era el modelo original a mercado; "
          f"~0.35/día era la revalidación pre-envío a precio de mercado, ya superada por este cambio)")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "min_net_rr": MIN_NET_RR, "max_wait_minutes": MAX_MINUTES_M15,
            "n_candidates_pre_retracement": len(sigs), "n_net_rr_ge_2": len(survived),
            "n_discarded_conflict": len(discarded_conflict), "n_discarded_concurrency": len(discarded_concurrency),
            "n_placed": n_placed, "n_filled": len(executed), "n_expired_unfilled": len(expired_unfilled),
            "n_days": n_days, "executed_per_day": len(executed) / n_days,
            "executed": executed, "expired_unfilled": expired_unfilled,
        }, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
