"""Revision diaria del piloto: analiza las operaciones CERRADAS de un dia y genera un
informe de diagnostico -- que simbolos, rangos de confianza y horas del dia funcionaron
mejor o peor. NO reentrena ni cambia config: es la misma clase de analisis que se hizo
a mano el 2026-08-24 al detectar el problema de stops pegados al spread, automatizada
para poder repetirla cada dia sin depender de que alguien la pida.

Deliberadamente NO alimenta directamente un reentrenamiento (ver memoria del proyecto:
entrenar sobre las propias operaciones del modelo tiene riesgo de sesgo de
retroalimentacion) -- el reentrenamiento real usa historico de mercado fresco
(ver scripts/weekly_retrain.py), este informe es solo para decidir ajustes de config
con criterio humano, igual que se hizo hoy con los stops y el umbral de confianza.

Uso:
    python scripts/daily_review.py                  # hoy (UTC)
    python scripts/daily_review.py --date 2026-08-24
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tradingai.mt5.trade_log import append_trade_event, COLUMNS as TRADE_LOG_COLUMNS  # noqa: E402

TRADES_LOG_DEFAULT = PROJECT_ROOT / "logs" / "live" / "trades.csv"
TIGHT_STOP_LOG_DEFAULT = PROJECT_ROOT / "logs" / "live" / "tight_stop_trades.csv"

# Distancia MINIMA de SL por debajo de la cual se considera "stop demasiado pegado"
# y se guarda para revision -- ver el hallazgo del 2026-08-25 (GBPUSD, SL a 7 pips,
# saltado en menos de 5 min). Son umbrales heuristicos en unidades de precio, no un
# analisis estadistico de volatilidad "normal" por simbolo; sirven para juntar casos
# candidatos a revisar, no para decidir por si solos si hay que corregir nada -- esa
# decision se toma con criterio humano cuando haya suficientes casos acumulados (ver
# memoria del proyecto: no reentrenar/ajustar riesgo en base a un caso aislado).
_JPY_MIN_SL = 0.08  # 8 pips (pip=0.01 en pares con JPY)
_FX_MIN_SL = 0.0008  # 8 pips (pip=0.0001 en el resto de pares forex)
MIN_SL_DISTANCE = {
    "USDJPY": _JPY_MIN_SL, "EURJPY": _JPY_MIN_SL, "GBPJPY": _JPY_MIN_SL,
    "XAUUSD": 2.0,   # oro: ~8 "pips" de 2 decimales
    "US500": 5.0,    # indice: puntos, no pips
}
_DEFAULT_MIN_SL = _FX_MIN_SL


def _min_sl_distance(symbol: str) -> float:
    return MIN_SL_DISTANCE.get(symbol, _DEFAULT_MIN_SL)


def _load_trades(path: Path, target_date: date_cls) -> tuple[dict[str, dict], dict[str, dict]]:
    opens: dict[str, dict] = {}
    closes: dict[str, dict] = {}
    if not path.exists():
        return opens, closes

    with open(path) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            if row["event"] == "APERTURA":
                opens[row["ticket"]] = row
            elif row["event"] == "CIERRE" and ts.date() == target_date:
                closes[row["ticket"]] = row
    return opens, closes


def _confidence_bucket(confidence: float) -> str:
    lo = int(confidence * 20) * 5
    return f"{lo}-{lo + 5}%"


def _print_breakdown(title: str, groups: dict[str, list[float]]) -> None:
    print(f"--- {title} ---")
    if not groups:
        print("  (sin datos)")
        return
    for key, profits in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        wins = sum(1 for p in profits if p > 0)
        print(f"  {key}: {wins}/{len(profits)} ganadas ({wins / len(profits) * 100:.0f}%), pnl=${sum(profits):.2f}")
    print()


def _already_flagged_tickets(tight_stop_log_path: Path) -> set[str]:
    if not tight_stop_log_path.exists():
        return set()
    with open(tight_stop_log_path) as f:
        return {row["ticket"] for row in csv.DictReader(f)}


def _flag_tight_stops(opens: dict[str, dict], target_date: date_cls, tight_stop_log_path: Path) -> int:
    already_flagged = _already_flagged_tickets(tight_stop_log_path)
    n_flagged = 0
    for ticket, open_row in opens.items():
        if ticket in already_flagged:
            continue
        if datetime.fromisoformat(open_row["timestamp"]).date() != target_date:
            continue
        entry, sl = float(open_row["price"] or 0), float(open_row["sl"] or 0)
        if not sl:
            continue
        if abs(entry - sl) < _min_sl_distance(open_row["symbol"]):
            append_trade_event(tight_stop_log_path, **{c: open_row.get(c, "") for c in TRADE_LOG_COLUMNS})
            n_flagged += 1
    return n_flagged


def run_review(target_date: date_cls, trades_log_path: Path, tight_stop_log_path: Path = TIGHT_STOP_LOG_DEFAULT) -> dict:
    opens, closes = _load_trades(trades_log_path, target_date)

    print(f"=== Revision diaria del piloto — {target_date.isoformat()} ===\n")

    n_tight = _flag_tight_stops(opens, target_date, tight_stop_log_path)
    if n_tight:
        print(f"Guardadas {n_tight} operacion(es) con stop demasiado pegado en {tight_stop_log_path}\n")

    if not closes:
        print("Sin operaciones cerradas ese dia.")
        return {"date": target_date.isoformat(), "n_closed": 0}

    by_symbol: dict[str, list[float]] = defaultdict(list)
    by_confidence: dict[str, list[float]] = defaultdict(list)
    by_hour: dict[str, list[float]] = defaultdict(list)
    sl_hits, tp_hits, unknown_close = 0, 0, 0

    total_pnl = 0.0
    wins = 0
    for ticket, close in closes.items():
        profit = float(close["profit"]) if close["profit"] not in ("", None) else 0.0
        total_pnl += profit
        if profit > 0:
            wins += 1

        symbol = close["symbol"]
        by_symbol[symbol].append(profit)

        open_row = opens.get(ticket)
        if open_row:
            confidence = float(open_row["confidence"])
            by_confidence[_confidence_bucket(confidence)].append(profit)
            open_ts = datetime.fromisoformat(open_row["timestamp"])
            by_hour[f"{open_ts.hour:02d}:00 UTC"].append(profit)

            # Heuristico simple: si el precio de cierre quedo mas cerca del SL que del
            # TP originales, lo contamos como "cierre por SL" (no es exacto al 100% si
            # hubo slippage, pero suficiente para ver el patron general).
            sl, tp, close_price = float(open_row["sl"]), float(open_row["tp"]), float(close["price"] or 0)
            if close_price and abs(close_price - sl) < abs(close_price - tp):
                sl_hits += 1
            else:
                tp_hits += 1
        else:
            unknown_close += 1

    n = len(closes)
    print(f"Operaciones cerradas: {n}")
    print(f"Win rate: {wins}/{n} ({wins / n * 100:.1f}%)")
    print(f"PnL total: ${total_pnl:.2f}  (medio: ${total_pnl / n:.2f}/operacion)")
    print(f"Cierres por SL (aprox.): {sl_hits}  |  por TP (aprox.): {tp_hits}  |  sin apertura registrada: {unknown_close}")
    print()

    _print_breakdown("Por simbolo", by_symbol)
    _print_breakdown("Por rango de confianza", by_confidence)
    _print_breakdown("Por hora UTC de apertura", by_hour)

    return {
        "date": target_date.isoformat(),
        "n_closed": n,
        "win_rate": wins / n,
        "total_pnl": total_pnl,
        "by_symbol": {k: sum(v) for k, v in by_symbol.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD en UTC; por defecto, hoy")
    parser.add_argument("--trades-log", default=None)
    parser.add_argument("--tight-stop-log", default=None)
    args = parser.parse_args()

    target_date = date_cls.fromisoformat(args.date) if args.date else datetime.now(timezone.utc).date()
    trades_log_path = Path(args.trades_log) if args.trades_log else TRADES_LOG_DEFAULT
    tight_stop_log_path = Path(args.tight_stop_log) if args.tight_stop_log else TIGHT_STOP_LOG_DEFAULT

    run_review(target_date, trades_log_path, tight_stop_log_path)


if __name__ == "__main__":
    main()
