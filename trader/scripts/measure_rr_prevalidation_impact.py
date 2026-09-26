"""Prompt (revalidación de R:R pre-envío): cuántas de las 225 señales que ya
ejecutaba el pipeline (`logs/mtf_frequency_final_with_costs.json` -- teórico
R:R>=2 -> neto de costo>=2 -> sin conflicto E1/E2 -> sin choque de concurrencia,
el mismo conjunto detrás de la frecuencia ~2.4 operaciones/día reportada) se
hubieran CANCELADO si además se revalida el R:R justo antes de enviar, contra
el precio de mercado real en vez del precio teórico de la señal.

PROXY de "precio al momento de enviar" (documentado, no inventado): la única
resolución histórica disponible es M15 (no hay tick ni M1 cacheados). En vivo,
`generated_at` es el cierre de la última vela M15 evaluada y la orden se manda
en el MISMO tick (segundos de latencia HTTP, no minutos) -- así que el precio
de mercado más cercano y disponible a "justo antes de enviar" es el CLOSE de
esa misma vela M15 (la vela cuyo cierre == generated_at). Es una cota INFERIOR
del slippage real (el poll real puede tardar hasta 5 min más en notar la vela),
así que el número de abajo es conservador -- probablemente subestima cuántas
señales se cancelarían en producción, no las sobreestima.

Reusa exactamente el mismo modelo de costo (`estimate_symbol_cost`/`cost_in_r`)
y el mismo `compute_trade_levels` que ya usa el gate en vivo -- ninguna
convención nueva.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import cost_in_r, estimate_symbol_cost
from trader.risk.levels import compute_trade_levels

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"
FREQ_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_frequency_final_with_costs.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_rr_prevalidation_impact.json"
MIN_NET_RR = 2.0


def _load_m15(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_M15.csv", parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df.sort_values("timestamp").set_index("timestamp")


def _price_at_send(m15_by_symbol: dict[str, pd.DataFrame], symbol: str, generated_at: pd.Timestamp) -> float | None:
    """Close de la vela M15 cuyo cierre coincide con generated_at (ver docstring
    del módulo). None si esa vela no está en el cache (borde de la ventana)."""
    bar_open = generated_at - pd.Timedelta(minutes=15)
    df = m15_by_symbol[symbol]
    if bar_open not in df.index:
        return None
    return float(df.loc[bar_open, "close"])


def main() -> None:
    with open(FREQ_PATH) as f:
        freq = json.load(f)
    with open(DATA_DIR / "symbol_info_full.json") as f:
        symbol_info = json.load(f)

    executed = freq["executed"]
    symbols = sorted({row["symbol"] for row in executed})
    m15_by_symbol = {s: _load_m15(s) for s in symbols}
    cost_by_symbol = {s: estimate_symbol_cost(s, symbol_info[s]["point"], m15_by_symbol[s].reset_index()) for s in symbols}

    rows = []
    for row in executed:
        symbol = row["symbol"]
        generated_at = pd.Timestamp(row["generated_at"])
        price_now = _price_at_send(m15_by_symbol, symbol, generated_at)

        out = dict(row)
        if price_now is None:
            out["outcome"] = "sin_dato_m15"  # borde de la ventana cacheada, no se pudo revalidar
            rows.append(out)
            continue

        levels_now = compute_trade_levels(row["direction"], price_now, row["sl"], row["tp2"], min_rr=0.0)
        out["price_at_send_proxy"] = price_now
        if levels_now is None:
            out["outcome"] = "cancelada_geometria_invalida"  # precio ya cruzó el SL o ya superó el TP2
            rows.append(out)
            continue

        risk_now = abs(price_now - row["sl"])
        net_rr_now = levels_now.risk_reward - cost_in_r(cost_by_symbol[symbol], risk_now)
        out["risk_reward_at_send"] = levels_now.risk_reward
        out["net_rr_at_send"] = net_rr_now
        out["outcome"] = "ejecutada" if net_rr_now >= MIN_NET_RR else "cancelada_rr_degradado"
        rows.append(out)

    df = pd.DataFrame(rows)
    n_total = len(df)
    n_no_data = int((df["outcome"] == "sin_dato_m15").sum())
    n_cancel_geom = int((df["outcome"] == "cancelada_geometria_invalida").sum())
    n_cancel_rr = int((df["outcome"] == "cancelada_rr_degradado").sum())
    n_still_executed = int((df["outcome"] == "ejecutada").sum())
    n_evaluable = n_total - n_no_data
    n_cancelled = n_cancel_geom + n_cancel_rr

    gen_at = pd.to_datetime(df["generated_at"])
    n_days = max((gen_at.max() - gen_at.min()).days, 1)
    rate_before = n_total / n_days
    rate_after = n_still_executed / n_days

    print("=== Impacto de la revalidación de R:R pre-envío (proxy: close de la vela M15 en generated_at) ===")
    print(f"Señales que ya pasaban el gate original (teórico->neto costo->conflicto->concurrencia): {n_total}")
    print(f"  sin dato M15 para revalidar (borde de ventana): {n_no_data}")
    print(f"  evaluables: {n_evaluable}")
    print(f"  canceladas por geometría inválida (precio ya cruzó SL o ya pasó TP2): {n_cancel_geom}")
    print(f"  canceladas por R:R degradado (< {MIN_NET_RR}): {n_cancel_rr}")
    print(f"  siguen ejecutándose: {n_still_executed}")
    print(f"\nTotal canceladas por la revalidación: {n_cancelled}/{n_evaluable} evaluables "
          f"({100*n_cancelled/n_evaluable:.1f}%)")
    print(f"\nFrecuencia ANTES (todo lo que pasaba el gate original): {rate_before:.2f} señales/día")
    print(f"Frecuencia DESPUÉS (sobrevive también la revalidación pre-envío): {rate_after:.2f} señales/día")
    print(f"Reducción: {100*(1 - rate_after/rate_before):.1f}%")

    print("\n=== desglose por estrategia ===")
    print(df.groupby(["strategy", "outcome"]).size().unstack(fill_value=0))

    with open(OUT_PATH, "w") as f:
        json.dump({
            "min_net_rr": MIN_NET_RR, "n_total": n_total, "n_no_data": n_no_data,
            "n_cancel_geometry": n_cancel_geom, "n_cancel_rr_degraded": n_cancel_rr,
            "n_still_executed": n_still_executed, "rate_before_per_day": rate_before,
            "rate_after_per_day": rate_after, "rows": rows,
        }, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
