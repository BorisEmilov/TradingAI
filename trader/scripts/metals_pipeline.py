"""Pipeline PREPARADO (no ejecutado) para sumar metales (XAUUSD, XAGUSD) al sistema MTF.
Diseño y condiciones: docs/metals_and_concurrency_design_2026-10-02.md.

BLOQUEADO hasta luz verde: cada subcomando (salvo `gate`) se niega a correr si
  1) logs/live_pilot_status.json no muestra n >= 50 en la serie de reglas vigentes
     sin criterio de abandono disparado (live_pilot_prereg_2026-10-02.md), y
  2) no existe logs/metals_greenlight.txt, que escribe el USUARIO a mano tras
     revisar la reevaluación de n=50 y aprobar el pre-registro de metales.
Si el piloto disparó el abandono, este plan se CANCELA (no se pospone).

Subcomandos (en orden, cada uno una sola vez):
  gate      solo informa si está habilitado
  fetch     D1/H1/M15 de XAUUSD/XAGUSD vía gateway: UN login, SIN logout (sesión
            compartida con el piloto). Reemplaza los CSV viejos de metales (bug de
            offset de fin de semana, sin H1) guardándolos como *.pre_offset_fix.csv
  spread    spread real de ticks por metal -> logs/metals_spread.json (sin logout)
  backtest  mismo harness que la OOS de FX (oos_prereg_2026_10_02), símbolos = metales,
            ventana y umbrales tomados del pre-registro de metales (args obligatorios)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
LOGS, DATA = ROOT / "logs", ROOT / "data" / "quant_battery"
STATUS, GREENLIGHT, SPREAD_OUT = LOGS / "live_pilot_status.json", LOGS / "metals_greenlight.txt", LOGS / "metals_spread.json"
METALS = ["XAUUSD", "XAGUSD"]
COUNTS = {"D1": 15000, "H1": 50000, "M15": 50000}
SPREAD_WINDOWS = {  # hora de SERVIDOR, sesión Londres 4h -- mismas que verify_spread_source + una reciente
    "pre_caida_2026-05-13": ("2026-05-13 10:00", "2026-05-13 14:00"),
    "reciente_2026-09-24": ("2026-09-24 10:00", "2026-09-24 14:00"),
}


def gate_status() -> tuple[bool, str]:
    if not STATUS.exists():
        return False, "sin logs/live_pilot_status.json -- el piloto todavía no reportó"
    crit = json.loads(STATUS.read_text())["criterio"]
    if crit.get("abandono"):
        return False, f"CANCELADO: el piloto disparó el criterio de abandono ({crit['abandono']})"
    if crit["n"] < crit.get("objetivo_n", 50):
        return False, f"bloqueado: piloto en n={crit['n']}/{crit.get('objetivo_n', 50)}"
    if not GREENLIGHT.exists():
        return False, f"bloqueado: n={crit['n']} alcanzado, falta la luz verde del usuario ({GREENLIGHT.name})"
    return True, f"habilitado: n={crit['n']}, sin abandono, luz verde presente"


def fetch(client) -> None:
    import pandas as pd
    print(f"offset servidor-UTC: {client.server_utc_offset()}")
    for symbol in METALS:
        dfs = {tf: client.candles(symbol, tf, n, timeout=120) for tf, n in COUNTS.items()}
        dow = dfs["M15"]["timestamp"].dt.dayofweek
        sat = int((dow == 5).sum())
        if sat:  # mismo chequeo del bug de offset que refetch_10_symbols
            raise SystemExit(f"{symbol}: {sat} velas M15 en sábado -> fechas corridas, no se guarda")
        for tf, df in dfs.items():
            path = DATA / f"{symbol}_{tf}.csv"
            if path.exists():
                path.rename(path.with_name(f"{symbol}_{tf}.pre_offset_fix.csv"))
            df.to_csv(path, index=False)
        print(f"{symbol}: M15 {dfs['M15']['timestamp'].iloc[0]} -> {dfs['M15']['timestamp'].iloc[-1]} guardado")


def spread(client) -> None:
    import pandas as pd
    from verify_spread_source import _compare
    out = {}
    for symbol in METALS:
        point = client.symbol_info(symbol, timeout=30)["point"]
        out[symbol] = {"point": point, **{w: _compare(client, symbol, point, int(pd.Timestamp(a, tz="UTC").timestamp()),
                                                     int(pd.Timestamp(b, tz="UTC").timestamp()))
                                          for w, (a, b) in SPREAD_WINDOWS.items()}}
    SPREAD_OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def backtest(scan_end: str, start: str, months_back: int, resolve_end: str) -> None:
    import pandas as pd
    import honest_reevaluation as hr
    import oos_prereg_2026_10_02 as oos
    sp = json.loads(SPREAD_OUT.read_text())
    # spread pesimista = mayor promedio de ticks medido (mismo criterio que FX), en precio con el point real
    hr._spread_price = lambda s: max(v["tick_spread_mean"] for k, v in sp[s].items() if k != "point") * sp[s]["point"]
    oos.SYMBOLS = METALS
    w = {"scan_end": pd.Timestamp(scan_end, tz="UTC"), "months_back": months_back,
         "start": pd.Timestamp(start, tz="UTC"), "resolve_end": pd.Timestamp(resolve_end, tz="UTC")}
    cand, n_unique, n_days = oos.candidates(w)
    res, frames = oos.evaluate(cand, w, n_unique, n_days)
    out = LOGS / f"metals_result_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.json"
    out.write_text(json.dumps({"window": {k: str(v) for k, v in w.items()}, "results": res,
                               "trades": {k: v.to_dict(orient="records") for k, v in frames.items()}}, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate"); sub.add_parser("fetch"); sub.add_parser("spread")
    bt = sub.add_parser("backtest")
    for a in ("--scan-end", "--start", "--resolve-end"):
        bt.add_argument(a, required=True)
    bt.add_argument("--months-back", type=int, required=True)
    args = ap.parse_args()

    ok, why = gate_status()
    print(why)
    if args.cmd == "gate" or not ok:
        raise SystemExit(0 if ok else 2)
    if args.cmd == "backtest":
        backtest(args.scan_end, args.start, args.months_back, args.resolve_end)
        return
    from trader.config import load_config
    from trader.gateway_client import PythonGetawayClient
    client = PythonGetawayClient(load_config().gateway)
    client.login()  # sin logout: la sesión es compartida con el piloto en vivo
    (fetch if args.cmd == "fetch" else spread)(client)


if __name__ == "__main__":
    main()
