"""Prompt "ampliar a 10 símbolos" (2026-09-25), punto 1: verificación PREVIA de
los pares candidatos antes de sumarlos a la config del piloto -- no asumir
nada, confirmar contra la API real.

Por candidato:
- trade_mode == 4 (FULL): el piloto opera long y short.
- Integridad del histórico D1/H1/M15: OHLC coherente, velas de rango 0,
  rachas de cierres idénticos (dato rellenado), huecos no-fin-de-semana, y
  CONSISTENCIA DE TIPO CRUZADO por época contra pares independientes (ej.
  EURJPY vs EURUSD*USDJPY) -- es lo que detecta dato sintético/erróneo en el
  tramo antiguo (mismo tipo de hallazgo que EURUSD pre-1999).
- Costo: spread real por vela (variable, no constante ni 0 -- un default
  silencioso sería constante) vía `estimate_symbol_cost`, contrastado con el
  spread en vivo.
- Sizing: `trade_tick_value` de MT5 vs valor INDEPENDIENTE
  (contract_size * tick_size * tasa moneda_cotización->USD, de otro par en
  vivo), y `compute_position_size_lots` vs lote calculado a mano.

Si un candidato falla, se prueba el siguiente de SUBSTITUTES hasta tener 7.

Un único login y logout al final (piloto parado, pool IDLE verificado antes
de correr -- ver project_gateway_shared_session_token_2026-09-23: NO correr
esto con el piloto vivo, el logout le tumbaría la sesión).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

import pandas as pd

from trader.backtest.costs import estimate_symbol_cost
from trader.config import load_config
from trader.gateway_client import PythonGetawayClient
from trader.mtf_strategies.session_risk import RISK_PCT_MIN, compute_position_size_lots

CANDIDATES = ["AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY"]
SUBSTITUTES = ["EURCHF", "EURAUD", "AUDJPY", "EURCAD", "CADJPY", "CHFJPY"]
N_NEEDED = 7
TIMEFRAMES = ("D1", "H1", "M15")
D1_USED_FROM = "2015-01-01"
MAX_REQUEST = 50000
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"
REPORT_PATH = OUT_DIR / "new_fx_pairs_verification.json"
SYMBOL_INFO_PATH = OUT_DIR / "symbol_info_full.json"

# D1 de pares independientes para el chequeo de tipo cruzado:
# symbol -> (fórmula sobre closes D1 del mismo día)
CROSS_CHECK = {
    "EURJPY": lambda d: d["EURUSD"] * d["USDJPY"],
    "EURGBP": lambda d: d["EURUSD"] / d["GBPUSD"],
    "GBPJPY": lambda d: d["GBPUSD"] * d["USDJPY"],
    "USDCHF": lambda d: d["EURCHF"] / d["EURUSD"],
    "USDCAD": lambda d: d["EURCAD"] / d["EURUSD"],
    "AUDUSD": lambda d: d["EURUSD"] / d["EURAUD"],
    "NZDUSD": lambda d: d["EURUSD"] / d["EURNZD"],
    "EURCHF": lambda d: d["EURUSD"] * d["USDCHF"],
    "EURAUD": lambda d: d["EURUSD"] / d["AUDUSD"],
    "EURCAD": lambda d: d["EURUSD"] * d["USDCAD"],
    "AUDJPY": lambda d: d["AUDUSD"] * d["USDJPY"],
    "CADJPY": lambda d: d["USDJPY"] / d["USDCAD"],
    "CHFJPY": lambda d: d["USDJPY"] / d["USDCHF"],
}
REFERENCE_D1 = ["EURUSD", "GBPUSD", "USDJPY", "EURCHF", "EURCAD", "EURAUD", "EURNZD", "AUDUSD", "USDCHF", "USDCAD"]
ERAS = [("<2000", None, "2000-01-01"), ("2000-2014", "2000-01-01", "2015-01-01"),
        ("2015-2023", "2015-01-01", "2024-01-01"), ("2024+", "2024-01-01", None)]
CROSS_TOL = 0.01  # >1% de desvío close-vs-cruce en un día = dato sospechoso (cierres a distinta hora dan ~0.1-0.3%)
CROSS_MAX_BAD_FRAC = 0.02  # tolerado en la época usada (2024+)

# moneda de cotización -> (par con USD, invertir)  para tasa XXX->USD
USD_CONVERSION = {"USD": None, "JPY": ("USDJPY", True), "CAD": ("USDCAD", True), "CHF": ("USDCHF", True),
                  "GBP": ("GBPUSD", False), "EUR": ("EURUSD", False), "AUD": ("AUDUSD", False), "NZD": ("NZDUSD", False)}


COUNTS = {"D1": 15000, "H1": 50000, "M15": 50000}


def _candles(client, symbol: str, tf: str) -> pd.DataFrame:
    """Precalentar con un pedido chico antes del grande: el terminal MT5 (Wine)
    se cayó con page fault (2026-09-25) al pedir 50000 D1 de un símbolo nunca
    sincronizado. Un 503 = worker caído -> abortar, no insistir."""
    for attempt in range(3):
        try:
            client.candles(symbol, tf, 200, timeout=60)
            time.sleep(3)
            return client.candles(symbol, tf, COUNTS[tf], timeout=120)
        except Exception as exc:  # noqa: BLE001
            if "503" in str(exc):
                raise SystemExit(f"worker MT5 caído (503) en {symbol} {tf} -- abortando") from exc
            print(f"  candles {symbol} {tf} intento {attempt + 1} falló: {exc}")
            time.sleep(15)
    raise RuntimeError(f"sin velas {symbol} {tf}")


def _integrity(df: pd.DataFrame, tf: str) -> dict:
    if len(df) == 0:
        return {"n": 0, "ok": False, "reason": "sin datos"}
    ohlc_bad = int(((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    zero_range = int((df["high"] - df["low"]).le(0).sum())
    # racha más larga de closes idénticos consecutivos
    same = df["close"].diff().eq(0)
    longest_flat = int(same.groupby((~same).cumsum()).sum().max())
    gaps = df["timestamp"].diff().dt.total_seconds() / 86400
    max_gap_days = float(gaps.max()) if len(df) > 1 else 0.0
    head = df.head(20)
    flat_spread_at_start = ("spread" in df.columns) and head["spread"].nunique() <= 1
    # velas de rango 0 sueltas aparecen en feriados/rollover; <1% tolerado
    ok = (ohlc_bad == 0 and longest_flat < (3 if tf == "D1" else 12) and max_gap_days < 5
          and zero_range / len(df) < 0.01)
    return {
        "n": len(df), "start": str(df["timestamp"].iloc[0].date()), "end": str(df["timestamp"].iloc[-1].date()),
        "ohlc_inconsistent": ohlc_bad, "zero_range_bars": zero_range, "longest_identical_close_run": longest_flat,
        "max_gap_days": round(max_gap_days, 2), "flat_spread_at_start": bool(flat_spread_at_start), "ok": bool(ok),
    }


def _cross_check(symbol: str, d1: pd.DataFrame, ref: dict[str, pd.DataFrame]) -> dict:
    closes = {s: df.set_index(df["timestamp"].dt.normalize())["close"] for s, df in ref.items()}
    closes[symbol] = d1.set_index(d1["timestamp"].dt.normalize())["close"]
    joined = pd.DataFrame(closes)
    try:
        implied = CROSS_CHECK[symbol](joined)
    except KeyError as exc:
        return {"ok_recent": False, "reason": f"falta par de referencia {exc}"}
    joined = joined.assign(_implied=implied).dropna(subset=[symbol, "_implied"])
    implied = joined["_implied"]
    dev = (joined[symbol] / implied - 1).abs()
    out = {}
    for name, lo, hi in ERAS:
        m = pd.Series(True, index=dev.index)
        if lo:
            m &= dev.index >= pd.Timestamp(lo, tz="UTC")
        if hi:
            m &= dev.index < pd.Timestamp(hi, tz="UTC")
        sub = dev[m]
        out[name] = None if len(sub) == 0 else {
            "n_days": len(sub), "median_dev_pct": round(100 * sub.median(), 3),
            "frac_days_dev_gt_1pct": round(float((sub > CROSS_TOL).mean()), 4), "max_dev_pct": round(100 * sub.max(), 2),
        }
    recent = out.get("2024+")
    out["ok_recent"] = bool(recent and recent["frac_days_dev_gt_1pct"] <= CROSS_MAX_BAD_FRAC)
    # primer año desde el cual TODO el resto de la serie es consistente (para acotar baterías futuras)
    bad_years = sorted({ts.year for ts in dev[dev > CROSS_TOL].index})
    yearly_bad = (dev > CROSS_TOL).groupby(dev.index.year).mean()
    dirty = yearly_bad[yearly_bad > CROSS_MAX_BAD_FRAC]
    out["clean_from_year"] = int(dirty.index.max()) + 1 if len(dirty) else int(dev.index.min().year)
    out["n_bad_years"] = len(bad_years)
    return out


def _sizing_check(info: dict, conv_rate: float) -> dict:
    expected_tick_value = info["trade_contract_size"] * info["trade_tick_size"] * conv_rate
    rel_err = abs(info["trade_tick_value"] / expected_tick_value - 1)
    pip = info["point"] * (10 if info["digits"] in (3, 5) else 1)
    entry = info["bid"]
    sl = entry - 30 * pip
    equity = 100_000.0
    lots = compute_position_size_lots(
        equity=equity, risk_pct=RISK_PCT_MIN, entry=entry, sl=sl,
        trade_tick_value=info["trade_tick_value"], trade_tick_size=info["trade_tick_size"],
        volume_min=info["volume_min"], volume_max=info["volume_max"], volume_step=info["volume_step"],
    )
    # a mano: USD en riesgo / (USD por lote en 30 pips), floor a 0.01
    usd_per_lot_30pips = info["trade_contract_size"] * 30 * pip * conv_rate
    hand_lots = int(equity * RISK_PCT_MIN / usd_per_lot_30pips * 100) / 100
    realized_risk_usd = lots * usd_per_lot_30pips
    return {
        "currency_profit": info["currency_profit"], "conv_rate_to_usd": conv_rate,
        "mt5_tick_value": info["trade_tick_value"], "independent_tick_value": expected_tick_value,
        "tick_value_rel_err": round(rel_err, 5),
        "lots_code": lots, "lots_hand": hand_lots, "realized_risk_usd": round(realized_risk_usd, 2),
        "ok": bool(rel_err < 0.01 and abs(lots - hand_lots) <= info["volume_step"] + 1e-9
                   and realized_risk_usd <= equity * RISK_PCT_MIN + 1e-6),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    try:
        account = client.account()
        print(f"=== login OK -- cuenta moneda={account.get('currency')} equity={account.get('equity')} ===")
        assert account.get("currency") == "USD", "conversión de sizing asume cuenta en USD"

        ref_d1 = {}
        for s in REFERENCE_D1:
            client.symbol_info(s, timeout=30)
            try:
                ref_d1[s] = _candles(client, s, "D1")
            except RuntimeError as exc:
                print(f"  referencia {s} no disponible: {exc}")
        conv_info = {pair: client.symbol_info(pair, timeout=30) for pair, _ in filter(None, USD_CONVERSION.values())}

        with open(SYMBOL_INFO_PATH) as f:
            symbol_info_full = json.load(f)
        report: dict = {"account_currency": account.get("currency"), "symbols": {}}
        accepted: list[str] = []

        for symbol in CANDIDATES + SUBSTITUTES:
            if len(accepted) >= N_NEEDED:
                break
            if symbol in SUBSTITUTES:
                print(f"\n(sustituto) ", end="")
            print(f"\n--- {symbol} ---")
            row: dict = {}
            report["symbols"][symbol] = row
            try:
                info = client.symbol_info(symbol, timeout=30)
            except Exception as exc:  # noqa: BLE001
                print(f"  symbol_info ERROR: {exc} -- RECHAZADO")
                row["symbol_info_error"] = str(exc)
                row["accepted"] = False
                continue

            tradable = info.get("trade_mode") == 4
            row["trade_mode"] = info.get("trade_mode")
            print(f"  trade_mode={info.get('trade_mode')} currency_profit={info.get('currency_profit')} digits={info.get('digits')}")

            try:
                dfs = {tf: _candles(client, symbol, tf) for tf in TIMEFRAMES}
            except RuntimeError as exc:
                row["candles_error"] = str(exc)
                row["accepted"] = False
                print(f"  => RECHAZADO ({exc})")
                continue
            for tf, df in dfs.items():  # guardar siempre: reevaluar después sin otro login
                df.to_csv(OUT_DIR / f"{symbol}_{tf}.csv", index=False)
            # D1 que DECIDE: desde 2015 (el piloto usa ~500 velas D1 = ~2 años). El
            # histórico completo se informa aparte -- el tramo 1970s-90s de este
            # bróker tiene velas de rango 0 / huecos largos (dato de relleno, mismo
            # patrón que EURUSD pre-1999), relevante para baterías largas, no acá.
            d1 = dfs["D1"]
            row["integrity"] = {
                "D1_full_history_info": _integrity(d1, "D1"),
                "D1": _integrity(d1[d1["timestamp"] >= pd.Timestamp(D1_USED_FROM, tz="UTC")].reset_index(drop=True), "D1"),
                "H1": _integrity(dfs["H1"], "H1"), "M15": _integrity(dfs["M15"], "M15"),
            }
            for tf, chk in row["integrity"].items():
                print(f"  {tf}: {chk}")
            row["cross_check"] = _cross_check(symbol, dfs["D1"], ref_d1)
            print(f"  cruce D1: {row['cross_check']}")
            integrity_ok = all(row["integrity"][tf]["ok"] for tf in TIMEFRAMES) and row["cross_check"]["ok_recent"]

            m15 = dfs["M15"]
            cost = estimate_symbol_cost(symbol, info["point"], m15)
            last3m = m15[m15["timestamp"] >= m15["timestamp"].iloc[-1] - pd.DateOffset(months=3)]
            cost_3m = estimate_symbol_cost(symbol, info["point"], last3m)
            pip = info["point"] * (10 if info["digits"] in (3, 5) else 1)
            real_spread = "spread" in m15.columns and m15["spread"].nunique() > 1 and cost.avg_spread_points > 0
            row["cost"] = {
                "avg_spread_points_2y": cost.avg_spread_points, "avg_spread_points_3m": cost_3m.avg_spread_points,
                "avg_spread_pips_3m": round(cost_3m.avg_spread_price / pip, 3), "live_spread_points": info.get("spread"),
                "n_distinct_spread_values": int(m15["spread"].nunique()) if "spread" in m15.columns else 0,
                "real_spread_data": bool(real_spread),
            }
            print(f"  costo: {row['cost']}")

            conv_spec = USD_CONVERSION.get(info["currency_profit"])
            if info["currency_profit"] not in USD_CONVERSION:
                row["sizing"] = {"ok": False, "reason": f"moneda {info['currency_profit']} sin tasa de conversión"}
            else:
                if conv_spec is None:
                    conv = 1.0
                else:
                    pair, invert = conv_spec
                    mid = (conv_info[pair]["bid"] + conv_info[pair]["ask"]) / 2
                    conv = 1 / mid if invert else mid
                row["sizing"] = _sizing_check(info, conv)
            print(f"  sizing: {row['sizing']}")

            accept = tradable and integrity_ok and real_spread and row["sizing"]["ok"]
            row["accepted"] = bool(accept)
            if accept:
                symbol_info_full[symbol] = info
                accepted.append(symbol)
                print("  => ACEPTADO")
            else:
                print("  => RECHAZADO")

        with open(SYMBOL_INFO_PATH, "w") as f:
            json.dump(symbol_info_full, f, indent=2, default=str)
        report["accepted"] = accepted
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n=== Aceptados ({len(accepted)}/{N_NEEDED}): {accepted} ===\nReporte: {REPORT_PATH}")
    finally:
        client.logout()
        print("=== logout ===")


if __name__ == "__main__":
    main()
