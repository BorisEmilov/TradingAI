"""prompt-inventario-real-instrumentos.md: real broker inventory, single clean
login (no login/logout cycling -- see the project's throttling incident).

1. List ALL symbol names (fast, names only).
2. Client-side regex categorization (metals, energy, indices, crypto, rates/
   bonds, plus the 3 known majors) -- the broker exposes 12000+ symbols total,
   mostly stocks/ETFs irrelevant here, so this narrows to a real shortlist
   before spending any per-symbol calls.
3. For the shortlist: full symbol_info (path/description/currency/trade_mode/
   swap -- confirms the regex guess and whether it's actually tradable), plus
   a depth+integrity check per timeframe (same synthetic-data check that
   caught EURUSD's pre-1999 D1 problem -- never assume history is clean).
4. Volume-type check: FX has no central clearing, so MT5 can only ever report
   tick volume for it -- confirmed by inspecting the RAW gateway response
   (bypassing gateway_client.candles(), which renames tick_volume->volume and
   drops real_volume) for `real_volume` across categories.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd
import requests

from trader.config import load_config
from trader.gateway_client import PythonGetawayClient

CATEGORY_PATTERNS = {
    "fx_major_known": re.compile(r"^(EURUSD|GBPUSD|USDJPY)$"),
    "metals": re.compile(r"^X(AU|AG|PT|PD)USD$"),
    "energy": re.compile(r"^(WTI|BRENT|UKOIL|USOIL|XTIUSD|XBRUSD|OIL)\b"),
    "indices": re.compile(r"^(US500|USTEC|US30|UK100|DE40|GER40|JPN225|NAS100|SPX500|UK100|FRA40|EU50|AUS200|HK50)$"),
    "crypto": re.compile(r"^(BTC|ETH|LTC|XRP|BCH|ADA|SOL|DOGE)USD"),
    "rates_bonds": re.compile(r"(BOND|GILT|BUND|TNOTE|TREAS|YIELD|T10Y|T2Y|T30Y|SCHATZ|BTP|OAT|JGB)", re.IGNORECASE),
    "other_fx": re.compile(r"^[A-Z]{6}$"),  # any other 6-letter FX-looking symbol, catch-all
}

TIMEFRAMES = ("D1", "H1", "M15")
DEPTH_COUNT = 50000


def categorize(name: str) -> str | None:
    for cat, pat in CATEGORY_PATTERNS.items():
        if pat.search(name):
            return cat
    return None


def raw_candles_with_real_volume(base_url: str, token: str, symbol: str, timeframe: str, count: int) -> list[dict]:
    """Bypass gateway_client.candles() (which drops real_volume) to inspect
    the field directly."""
    resp = requests.get(
        f"{base_url}/market/candles/{symbol}",
        params={"timeframe": timeframe, "count": count, "include_current": False},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["candles"]


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    session = client.login()
    print("=== login OK, single session for the whole inventory ===")

    report: dict = {"categories": {}, "specs": {}, "depth": {}, "volume_check": {}}

    try:
        all_names = client.list_symbols()
        print(f"Total simbolos en el broker: {len(all_names)}")

        shortlist: dict[str, list[str]] = {}
        for name in all_names:
            cat = categorize(name)
            if cat and cat != "other_fx":
                shortlist.setdefault(cat, []).append(name)
        # other_fx: cap it, there could be dozens of legit crosses -- not the focus here
        other_fx = sorted({n for n in all_names if CATEGORY_PATTERNS["other_fx"].search(n) and categorize(n) == "other_fx"})
        print(f"\nOtros pares FX de 6 letras detectados (no profundizados, solo listados): {len(other_fx)}")
        print(" ", other_fx[:40], "..." if len(other_fx) > 40 else "")

        print("\n=== CANDIDATOS POR CATEGORIA (regex sobre nombres) ===")
        for cat, names in shortlist.items():
            print(f"{cat}: {names}")
        report["categories"] = shortlist
        report["other_fx_count"] = len(other_fx)
        report["other_fx_sample"] = other_fx[:40]

        # dedupe + flatten shortlist for deep-check, majors already well-known -- skip re-checking those
        deep_check = sorted({n for names in shortlist.values() for n in names if n not in ("EURUSD", "GBPUSD", "USDJPY")})
        print(f"\n=== VERIFICACION PROFUNDA: {len(deep_check)} simbolos ===")

        for symbol in deep_check:
            try:
                info = client.symbol_info(symbol, timeout=30)
            except Exception as exc:  # noqa: BLE001
                print(f"{symbol}: symbol_info ERROR -- {exc}")
                continue
            report["specs"][symbol] = {
                "path": info.get("path"), "description": info.get("description"),
                "currency_base": info.get("currency_base"), "currency_profit": info.get("currency_profit"),
                "trade_mode": info.get("trade_mode"), "swap_mode": info.get("swap_mode"),
                "point": info.get("point"), "digits": info.get("digits"),
            }
            print(f"\n{symbol}: path={info.get('path')!r} desc={info.get('description')!r} "
                  f"trade_mode={info.get('trade_mode')} swap_mode={info.get('swap_mode')}")

            depth_row = {}
            for tf in TIMEFRAMES:
                try:
                    df = client.candles(symbol, tf, DEPTH_COUNT, timeout=120)
                except Exception as exc:  # noqa: BLE001
                    body = getattr(getattr(exc, "response", None), "text", str(exc))
                    print(f"  {tf}: ERROR -- {body[:200]}")
                    depth_row[tf] = None
                    continue
                if len(df) == 0:
                    print(f"  {tf}: sin datos")
                    depth_row[tf] = {"n": 0}
                    continue
                start, end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
                span_years = (end - start).days / 365.25
                # cheap synthetic-data smell test: flat spread + suspiciously round volume in the first 20 rows
                head = df.head(20)
                flat_spread = ("spread" in df.columns) and (head["spread"].nunique() <= 1)
                depth_row[tf] = {
                    "n": len(df), "start": str(start.date()), "end": str(end.date()),
                    "years": round(span_years, 2), "flat_spread_at_start": bool(flat_spread),
                }
                flag = " <-- spread plano al inicio, revisar integridad" if flat_spread else ""
                print(f"  {tf}: {len(df)} velas, {start.date()} -> {end.date()} (~{span_years:.2f} anios){flag}")
            report["depth"][symbol] = depth_row

        # volume-type check: raw candles (bypassing the client's column-dropping) for one representative
        # per category, to confirm real_volume is genuinely absent (FX/CFD synthetic) vs present (real feed)
        print("\n=== CHEQUEO DE TIPO DE VOLUMEN (real_volume crudo, no filtrado por el cliente) ===")
        probe_symbols = ["EURUSD"] + [names[0] for cat, names in shortlist.items() if names and cat != "other_fx"]
        for symbol in dict.fromkeys(probe_symbols):  # dedupe, keep order
            try:
                raw = raw_candles_with_real_volume(config.gateway.base_url, session.token, symbol, "D1", 20)
            except Exception as exc:  # noqa: BLE001
                print(f"{symbol}: ERROR -- {exc}")
                continue
            if not raw:
                print(f"{symbol}: sin velas")
                continue
            sample = raw[-1]
            tick_vol = sample.get("tick_volume")
            real_vol = sample.get("real_volume")
            report["volume_check"][symbol] = {"tick_volume": tick_vol, "real_volume": real_vol}
            print(f"{symbol}: tick_volume={tick_vol}, real_volume={real_vol} {'(FX/CFD sintetico -- solo tick)' if not real_vol else '(VOLUMEN REAL DISPONIBLE)'}")

    finally:
        client.logout()
        print("\n=== logout OK ===")

    out_path = Path(__file__).resolve().parent.parent / "data" / "quant_battery" / "instrument_inventory.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReporte guardado en {out_path}")


if __name__ == "__main__":
    main()
