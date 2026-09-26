"""Prompt "2 verificaciones antes de relanzar con 10 símbolos" (2026-09-25),
punto 2: ¿el campo `spread` de vela (~0 en 97-99% de las M15 desde jul-2026)
refleja el spread REAL, o es un artefacto de cómo MT5 guarda ese campo?

Compara, vela a vela, el campo `spread` de la M15 contra el spread bid-ask
real de los ticks de ESA MISMA vela (`/market/ticks/{sym}/range`, mismo epoch
de servidor que `/market/candles/{sym}/range` -> se alinean por
floor(time/900)*900), en dos ventanas: una reciente (después de la caída a 0)
y una de mayo-2026 (antes). Además muestrea el tick en vivo (`/market/ticks`).

Un login, un logout. Solo lectura. Piloto parado (verificar pool IDLE antes).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY"]
# ventanas en HORA DE SERVIDOR (epoch que esperan ambos endpoints): sesión Londres, 4h
WINDOWS = {
    "reciente_2026-09-24": ("2026-09-24 10:00", "2026-09-24 14:00"),
    "pre_caida_2026-05-13": ("2026-05-13 10:00", "2026-05-13 14:00"),
}
LIVE_SAMPLES, LIVE_EVERY_S = 6, 20
OUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "spread_source_verification.json"


def _epoch(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def _compare(client, symbol: str, point: float, frm: int, to: int) -> dict:
    candles = pd.DataFrame(client._request("GET", f"/market/candles/{symbol}/range", timeout=60,
                                           params={"timeframe": "M15", "from_epoch": frm, "to_epoch": to})["candles"])
    ticks = pd.DataFrame(client._request("GET", f"/market/ticks/{symbol}/range", timeout=120,
                                         params={"from_epoch": frm, "to_epoch": to, "flags": "ALL"})["ticks"])
    if candles.empty or ticks.empty:
        return {"error": f"sin datos (velas={len(candles)}, ticks={len(ticks)})"}
    ticks = ticks[(ticks["bid"] > 0) & (ticks["ask"] > 0)]
    ticks["bar"] = (ticks["time"] // 900) * 900
    ticks["spread_pts"] = ((ticks["ask"] - ticks["bid"]) / point).round()
    per_bar = ticks.groupby("bar")["spread_pts"].agg(tick_min="min", tick_median="median", tick_mean="mean")
    j = candles.set_index("time")[["spread"]].join(per_bar, how="inner")
    return {
        "n_bars": len(j), "n_ticks": len(ticks),
        "candle_field_mean": round(j["spread"].mean(), 2),
        "candle_field_zero_pct": round(100 * (j["spread"] == 0).mean(), 1),
        "tick_spread_mean": round(j["tick_mean"].mean(), 2),
        "tick_spread_median": round(j["tick_median"].median(), 2),
        "tick_spread_min_mean": round(j["tick_min"].mean(), 2),
        "candle_field_equals_tick_min_pct": round(100 * (j["spread"] == j["tick_min"]).mean(), 1),
    }


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    report: dict = {"windows": {}, "live": {}}
    try:
        points = {s: client.symbol_info(s, timeout=30)["point"] for s in SYMBOLS}
        for label, (a, b) in WINDOWS.items():
            report["windows"][label] = {}
            print(f"\n=== {label} (vela M15 vs ticks bid-ask de la misma vela, en puntos) ===")
            for s in SYMBOLS:
                try:
                    r = _compare(client, s, points[s], _epoch(a), _epoch(b))
                except Exception as exc:  # noqa: BLE001
                    r = {"error": str(exc)}
                report["windows"][label][s] = r
                print(f"  {s}: {r}")

        print(f"\n=== tick en vivo, {LIVE_SAMPLES} muestras cada {LIVE_EVERY_S}s (spread_points = ask-bid) ===")
        samples = {s: [] for s in SYMBOLS}
        for i in range(LIVE_SAMPLES):
            data = client._request("GET", "/market/ticks", params={"symbols": ",".join(SYMBOLS)})
            for s, t in data["ticks"].items():
                samples[s].append(t.get("spread_points"))
            if i < LIVE_SAMPLES - 1:
                time.sleep(LIVE_EVERY_S)
        for s in SYMBOLS:
            report["live"][s] = samples[s]
            print(f"  {s}: {samples[s]}")
    finally:
        client.logout()
        print("=== logout ===")

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
