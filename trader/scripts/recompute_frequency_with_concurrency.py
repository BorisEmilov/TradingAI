"""Prompt 5: recalcula la frecuencia real aplicando, en orden: gate R:R>=2
(ya aplicado -- parte de `signals_unique`) -> conflicto E1/E2 por símbolo ->
concurrencia por par (no límite global de sesión). Concurrencia es
independiente ENTRE símbolos (una posición en EURUSD no bloquea GBPUSD), así
que se procesa símbolo por símbolo, cronológicamente -- más simple y
exactamente equivalente a una simulación de eventos cruzada.

Aproximación declarada: el momento de decisión para concurrencia/conflicto es
`generated_at` (la PRIMERA vela en la que la señal se detectó, ya
deduplicada en el escaneo del Paso 3) -- no se vuelve a escanear la lista
cruda de 7702 chequeos de vela. Un conflicto que solo se daría en una
redetección posterior (no en la primera detección) no se capturaría acá;
dado que ambas estrategias completan su secuencia entera para disparar,
coincidir en la PRIMERA detección exacta ya es el caso más común de
solapamiento real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.mtf_strategies.signal import MTFSignal
from trader.mtf_strategies.session_risk import PositionConcurrencyState, can_open_new_trade, record_position_closed, record_position_opened
from trader.mtf_strategies.trade_simulation import simulate_position_close

RESULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_scan_result.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"
OUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_frequency_final.json"


def _load_m15(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_M15.csv", parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df.sort_values("timestamp").reset_index(drop=True)


def _to_signal(row: dict) -> MTFSignal:
    return MTFSignal(
        strategy=row["strategy"], setup=row["setup"], symbol=row["symbol"], direction=row["direction"],
        session=row["session"], entry=row["entry"], sl=row["sl"], tp1=row["tp1"], tp2=row["tp2"],
        risk_reward=row["risk_reward"], sweep_timestamp=pd.Timestamp(row["sweep_timestamp"]),
        mss_timestamp=pd.Timestamp(row["mss_timestamp"]), fvg_confirmed_at=pd.Timestamp(row["fvg_confirmed_at"]),
        generated_at=pd.Timestamp(row["generated_at"]), trace=row.get("trace", {}),
    )


def main() -> None:
    with open(RESULT_PATH) as f:
        scan = json.load(f)
    sigs = pd.DataFrame(scan["signals_unique"])
    sigs["generated_at"] = pd.to_datetime(sigs["generated_at"])

    executed: list[dict] = []
    discarded_conflict: list[dict] = []
    discarded_concurrency: list[dict] = []

    for symbol in sorted(sigs["symbol"].unique()):
        m15 = _load_m15(symbol)
        sym_sigs = sigs[sigs["symbol"] == symbol].sort_values("generated_at").reset_index(drop=True)

        state = PositionConcurrencyState()
        open_until: pd.Timestamp | None = None  # cuándo se libera `symbol`, según la posición abierta actual (si hay)

        # paso 1: conflicto E1/E2 -- mismo símbolo, mismo generated_at exacto, ambas estrategias
        moment_counts = sym_sigs.groupby("generated_at")["strategy"].apply(lambda s: set(s))
        conflicted_moments = {t for t, strategies in moment_counts.items() if len(strategies) > 1}

        for _, row in sym_sigs.iterrows():
            if row["generated_at"] in conflicted_moments:
                discarded_conflict.append(row.to_dict())
                continue

            # si la posición previamente abierta en este símbolo ya cerró para este momento, liberarlo
            if open_until is not None and row["generated_at"] >= open_until:
                state = record_position_closed(state, symbol)
                open_until = None

            # paso 2: concurrencia por símbolo -- usa la API real, no una comparación de timestamps aparte
            ok, _reason = can_open_new_trade(state, symbol)
            if not ok:
                discarded_concurrency.append(row.to_dict())
                continue

            # ejecutable: simular hasta el cierre para saber cuándo el símbolo vuelve a estar libre
            sig = _to_signal(row.to_dict())
            close_ts = simulate_position_close(sig, m15)
            row_dict = row.to_dict()
            row_dict["simulated_close_at"] = str(close_ts) if close_ts is not None else None
            executed.append(row_dict)
            state = record_position_opened(state, symbol)
            open_until = close_ts if close_ts is not None else pd.Timestamp.max.tz_localize("UTC")

    exec_df = pd.DataFrame(executed)
    print(f"=== RESULTADO FINAL (gate R:R>=2 -> conflicto E1/E2 por simbolo -> concurrencia por par) ===")
    print(f"Setups candidatos (post R:R>=2, pre conflicto/concurrencia): {len(sigs)}")
    print(f"Descartados por conflicto E1/E2 (mismo simbolo, mismo momento): {len(discarded_conflict)}")
    print(f"Descartados por concurrencia (posicion ya abierta en el par): {len(discarded_concurrency)}")
    print(f"EJECUTADOS: {len(exec_df)}")
    print()
    print("Por simbolo x estrategia:")
    print(exec_df.groupby(["symbol", "strategy"]).size().unstack(fill_value=0))
    print()
    n_weeks = (sigs["generated_at"].max() - sigs["generated_at"].min()).days / 7
    n_days = (sigs["generated_at"].max() - sigs["generated_at"].min()).days
    print(f"Ventana ~{n_weeks:.1f} semanas (~{n_days} dias)")
    print(f"Promedio ejecutadas/semana (los 3 simbolos): {len(exec_df)/n_weeks:.2f}")
    print(f"Promedio ejecutadas/dia (los 3 simbolos): {len(exec_df)/n_days:.2f}")
    print()
    print("Por simbolo, ejecutadas/semana:")
    print((exec_df.groupby("symbol").size() / n_weeks).round(2))

    with open(OUT_PATH, "w") as f:
        json.dump({
            "executed": executed, "discarded_conflict": discarded_conflict, "discarded_concurrency": discarded_concurrency,
            "n_candidates": len(sigs), "n_executed": len(exec_df),
        }, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
