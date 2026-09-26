"""Paso 3, punto 3: auditoría visual (textual -- no hay kaleido/matplotlib
instalado, plotly HTML no se puede "mirar" desde acá tampoco -- verificación
numérica directa contra las velas crudas, mismo principio que un chequeo
visual: confirmar que el sweep, el MSS, la FVG y el TP2 tienen sentido).

Toma un muestreo de hasta 10 señales ÚNICAS (deduplicadas) por estrategia
desde logs/mtf_scan_result.json y para cada una imprime las velas 1H
alrededor del sweep, la vela M15 del MSS, la FVG, y el nivel de TP2
comparado contra el rango de precio reciente -- para verificar a mano que
cada pieza es real, no un artefacto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

RESULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_scan_result.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"


def _load(symbol: str, tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_{tf}.csv", parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df.sort_values("timestamp").reset_index(drop=True)


def audit_one(sig: dict, h1_cache: dict, m15_cache: dict) -> dict:
    symbol = sig["symbol"]
    if symbol not in h1_cache:
        h1_cache[symbol] = _load(symbol, "H1")
        m15_cache[symbol] = _load(symbol, "M15")
    h1, m15 = h1_cache[symbol], m15_cache[symbol]

    sweep_ts = pd.Timestamp(sig["sweep_timestamp"])
    mss_ts = pd.Timestamp(sig["mss_timestamp"])
    fvg_ts = pd.Timestamp(sig["fvg_confirmed_at"])

    print(f"\n{'='*90}\n{sig['strategy']} {sig.get('setup')} {symbol} {sig['direction']} @ {sig['generated_at']}")
    print(f"  entry={sig['entry']:.5f} sl={sig['sl']:.5f} tp1={sig['tp1']:.5f} tp2={sig['tp2']:.5f} rr={sig['risk_reward']:.2f}")
    print(f"  trace: {sig['trace']}")

    # -- sweep: la vela 1H cuyo close_ts == sweep_ts --
    sweep_row = h1[h1["timestamp"] + pd.Timedelta(hours=1) == sweep_ts]
    if len(sweep_row) == 1:
        row = sweep_row.iloc[0]
        level = sig["trace"].get("liquidity_level_price") or sig["trace"].get("liquidity_price") or sig["trace"].get("htf_level_price")
        print(f"  SWEEP 1H @ {row['timestamp']}: O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f} (nivel barrido~{level})")
        ok_sweep = True
        if level is not None:
            if sig["direction"] == "long":
                ok_sweep = row["low"] < level < row["close"]
            else:
                ok_sweep = row["high"] > level > row["close"]
        print(f"    -> mecha rompe el nivel y cierra de vuelta: {'OK' if ok_sweep else 'REVISAR'}")
    else:
        print(f"  SWEEP 1H: vela no encontrada para {sweep_ts} (n={len(sweep_row)})")
        ok_sweep = False

    # -- MSS: la vela M15 cuyo close_ts == mss_ts --
    mss_row = m15[m15["timestamp"] + pd.Timedelta(minutes=15) == mss_ts]
    if len(mss_row) == 1:
        row = mss_row.iloc[0]
        print(f"  MSS 15M @ {row['timestamp']}: O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f}")
        print(f"    -> posterior al sweep ({mss_ts} > {sweep_ts}): {'OK' if mss_ts > sweep_ts else 'REVISAR'}")
    else:
        print(f"  MSS 15M: vela no encontrada para {mss_ts}")

    # -- FVG: 3 velas M15 terminando en fvg_ts --
    fvg_end_idx = m15.index[m15["timestamp"] + pd.Timedelta(minutes=15) == fvg_ts]
    if len(fvg_end_idx) == 1:
        i = fvg_end_idx[0]
        c1, c2, c3 = m15.iloc[i - 2], m15.iloc[i - 1], m15.iloc[i]
        if sig["direction"] == "long":
            gap_ok = c1["high"] < c3["low"]
            gap = (c1["high"], c3["low"])
        else:
            gap_ok = c3["high"] < c1["low"]
            gap = (c3["high"], c1["low"])
        print(f"  FVG 15M (C1={c1['timestamp']}, C3={c3['timestamp']}): gap={gap} genuino={'OK' if gap_ok else 'REVISAR'}")
        entry_in_gap = min(gap) <= sig["entry"] <= max(gap)
        print(f"    -> entry (50%) dentro del rango del FVG: {'OK' if entry_in_gap else 'REVISAR'}")
    else:
        print(f"  FVG 15M: no se pudo ubicar C3 en {fvg_ts}")

    # -- TP2: comparar contra el rango de precio de las ultimas 200 velas H1 --
    recent_h1 = h1[h1["timestamp"] <= sweep_ts].tail(200)
    if len(recent_h1):
        lo, hi = recent_h1["low"].min(), recent_h1["high"].max()
        tp2_in_range = lo * 0.98 <= sig["tp2"] <= hi * 1.02  # holgura del 2% -- PWH/4H puede quedar apenas fuera del rango de 200 H1
        print(f"  TP2={sig['tp2']:.5f} vs rango reciente 1H [{lo:.5f}, {hi:.5f}]: {'OK (dentro de rango razonable)' if tp2_in_range else 'REVISAR (fuera de rango)'}")

    return {"sweep_ok": bool(ok_sweep)}


def main() -> None:
    with open(RESULT_PATH) as f:
        data = json.load(f)

    signals = data["signals_unique"]
    by_strategy: dict[str, list[dict]] = {}
    for s in signals:
        by_strategy.setdefault(s["strategy"], []).append(s)

    h1_cache: dict = {}
    m15_cache: dict = {}
    for strategy, sigs in by_strategy.items():
        print(f"\n\n{'#'*90}\n# {strategy}: {len(sigs)} señales únicas totales -- auditando hasta 10\n{'#'*90}")
        sample = sigs[:: max(1, len(sigs) // 10)][:10]  # muestreo espaciado, no solo las primeras
        if not sample:
            print("  (ninguna señal generada por esta estrategia en la ventana -- se reporta explícitamente, no se fuerza el número)")
            continue
        print(f"  auditando {len(sample)} de {len(sigs)}")
        for sig in sample:
            audit_one(sig, h1_cache, m15_cache)


if __name__ == "__main__":
    main()
