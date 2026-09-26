"""Prompt 6, punto 1: ni el gate de R:R>=2 (`risk/levels.py::compute_trade_levels`,
llamado desde `continuation.py`/`reversal.py`) ni `trade_simulation.py` descontaban
el spread real -- confirmado con `grep -i "cost\\|spread"` sobre los 4 archivos
del pipeline E1/E2, cero resultados antes de este script.

Recalcula el R:R neto de costo reusando el mecanismo YA VALIDADO del
proyecto (`trader.backtest.costs.estimate_symbol_cost`/`cost_in_r` -- no una
convención nueva): `net_rr = risk_reward_teorico - cost_in_r`, misma
identidad que ya usa `backtest/trade.py` para pasar de `gross_r` a `net_r`
(un costo de spread ida-y-vuelta, expresado en unidades de R, restado una
sola vez -- no se ajustan los precios de entrada/SL/TP de la simulación de
barra a barra, que sigue siendo la misma que ya está validada; el costo es
una capa de contabilidad aparte, igual que en el resto del proyecto).

Filtra los 299 candidatos (R:R teórico >=2) a los que además cumplen
R:R NETO >=2, y vuelve a correr exactamente la misma cadena de conflicto
E1/E2 + concurrencia por símbolo del Prompt 5 sobre el conjunto resultante.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import cost_in_r, estimate_symbol_cost
from trader.mtf_strategies.session_risk import PositionConcurrencyState, can_open_new_trade, record_position_closed, record_position_opened
from trader.mtf_strategies.signal import MTFSignal
from trader.mtf_strategies.trade_simulation import simulate_position_close

RESULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_scan_result.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"
OUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_frequency_final_with_costs.json"


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
    with open(DATA_DIR / "symbol_info_full.json") as f:
        symbol_info = json.load(f)

    sigs = pd.DataFrame(scan["signals_unique"])
    sigs["generated_at"] = pd.to_datetime(sigs["generated_at"])

    m15_cache: dict[str, pd.DataFrame] = {}
    symbol_cost_cache: dict = {}
    for symbol in sigs["symbol"].unique():
        m15_cache[symbol] = _load_m15(symbol)
        symbol_cost_cache[symbol] = estimate_symbol_cost(symbol, symbol_info[symbol]["point"], m15_cache[symbol])

    net_rr, cost_r_col = [], []
    for _, row in sigs.iterrows():
        risk_price = abs(row["entry"] - row["sl"])
        c_r = cost_in_r(symbol_cost_cache[row["symbol"]], risk_price)
        cost_r_col.append(c_r)
        net_rr.append(row["risk_reward"] - c_r)
    sigs["cost_r"] = cost_r_col
    sigs["net_rr"] = net_rr

    print("=== costo real de spread, por símbolo (round-trip, en R) ===")
    for symbol, cost in symbol_cost_cache.items():
        print(f"  {symbol}: avg_spread_points={cost.avg_spread_points:.2f} avg_spread_price={cost.avg_spread_price:.6f}")

    before = len(sigs)
    survived = sigs[sigs["net_rr"] >= 2.0].reset_index(drop=True)
    dropped = sigs[sigs["net_rr"] < 2.0]
    print(f"\nCandidatos con R:R TEORICO >=2: {before}")
    print(f"Candidatos que TAMBIEN cumplen R:R NETO (de costo real) >=2: {len(survived)}")
    print(f"Descartados solo por el costo (R:R teorico>=2 pero neto<2): {len(dropped)}")
    if len(dropped):
        print("  distribucion de cuanto les faltaba (2.0 - net_rr):")
        print((2.0 - dropped["net_rr"]).describe())

    # -- re-correr conflicto E1/E2 + concurrencia por simbolo sobre el conjunto ya filtrado por costo --
    executed, discarded_conflict, discarded_concurrency = [], [], []
    for symbol in sorted(survived["symbol"].unique()):
        m15 = m15_cache[symbol]
        sym_sigs = survived[survived["symbol"] == symbol].sort_values("generated_at").reset_index(drop=True)
        state = PositionConcurrencyState()
        open_until = None
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
            close_ts = simulate_position_close(sig, m15)
            row_dict = row.to_dict()
            row_dict["simulated_close_at"] = str(close_ts) if close_ts is not None else None
            executed.append(row_dict)
            state = record_position_opened(state, symbol)
            open_until = close_ts if close_ts is not None else pd.Timestamp.max.tz_localize("UTC")

    exec_df = pd.DataFrame(executed)
    print(f"\n=== RESULTADO FINAL CON COSTO REAL (R:R neto>=2 -> conflicto E1/E2 -> concurrencia) ===")
    print(f"Descartados por conflicto E1/E2: {len(discarded_conflict)}")
    print(f"Descartados por concurrencia: {len(discarded_concurrency)}")
    print(f"EJECUTADOS: {len(exec_df)} (vs 229 sin descontar costo)")
    if len(exec_df):
        print(exec_df.groupby(["symbol", "strategy"]).size().unstack(fill_value=0))
        n_weeks = (survived["generated_at"].max() - survived["generated_at"].min()).days / 7
        print(f"\nPromedio ejecutadas/semana (los 3 simbolos): {len(exec_df)/n_weeks:.2f}")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "n_theoretical_rr_ge_2": before, "n_net_rr_ge_2": len(survived),
            "n_dropped_by_cost": len(dropped), "executed": executed,
            "n_executed": len(exec_df),
        }, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
