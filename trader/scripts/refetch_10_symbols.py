"""Re-descarga D1/H1/M15 de los 10 símbolos del piloto MTF (2026-09-25) tras
encontrar que los CSV de EURUSD/GBPUSD/USDJPY bajados el sábado 2026-09-19
estaban corridos +23h (offset de servidor medido con un tick viejo -- ver
`PythonGetawayClient.server_utc_offset`). Correr SOLO con mercado abierto
(el cliente ahora falla en voz alta si no puede medir el offset).

Chequeo post-descarga: ninguna vela M15 en sábado, y apertura del domingo
entre 20:00 y 23:59 UTC. Sin logout (el piloto en vivo comparte el token).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_mtf_pilot import SYMBOLS
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

COUNTS = {"D1": 15000, "H1": 50000, "M15": 50000}
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"


def main() -> None:
    client = PythonGetawayClient(load_config().gateway)
    client.login()
    print(f"offset servidor-UTC: {client.server_utc_offset()}")
    for symbol in SYMBOLS:
        dfs = {tf: client.candles(symbol, tf, n, timeout=120) for tf, n in COUNTS.items()}
        m15 = dfs["M15"]
        dow = m15["timestamp"].dt.dayofweek
        sat = int((dow == 5).sum())
        sun_hours = sorted(m15.loc[dow == 6, "timestamp"].dt.hour.unique().tolist())
        ok = sat == 0 and sun_hours and min(sun_hours) >= 20
        print(f"{symbol}: M15 {m15['timestamp'].iloc[0]} -> {m15['timestamp'].iloc[-1]} | velas sábado={sat} "
              f"horas domingo={sun_hours} -> {'OK' if ok else 'FALLA'}")
        if not ok:
            raise SystemExit(f"{symbol}: fechas sospechosas, no se guarda")
        for tf, df in dfs.items():
            df.to_csv(OUT_DIR / f"{symbol}_{tf}.csv", index=False)
    print("sin logout (sesión compartida con el piloto en vivo)")


if __name__ == "__main__":
    main()
