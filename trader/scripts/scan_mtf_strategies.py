"""Paso 3: escanea una ventana histórica real (no la ventana chica del
smoke-test) evaluando Estrategia 1 y Estrategia 2 en cada M15 dentro de las
ventanas de entrada (Londres/NY 08-11 local), para los 3 majors cacheados.
Usa `PrecomputedHistory` (analysis.py) -- computa cada detector UNA VEZ por
símbolo/timeframe, slice barato por cutoff -- ya verificado idéntico a
`build_analysis()` recompute-desde-cero en 8 cutoffs aleatorios.

Guarda TODAS las señales (antes y después del gate de R:R -- el gate ya
corre adentro de evaluate_continuation/evaluate_reversal, así que "antes de
R:R" acá significa "el nivel candidato llegó hasta el punto de necesitar un
TP2/R:R pero no lo consiguió", extraído de NoSignal.reason) a JSON para
frecuencia + auditoría visual + re-chequeo de R:R.

Solo cuenta/registra -- no calcula expectancy ni nada parecido a la batería
estadística de fases anteriores (explícitamente fuera de alcance de este
paso).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.config import load_config
from trader.mtf_strategies.analysis import PrecomputedHistory
from trader.mtf_strategies.continuation import evaluate_continuation
from trader.mtf_strategies.reversal import evaluate_reversal
from trader.mtf_strategies.session_risk import active_entry_session
from trader.mtf_strategies.signal import MTFSignal, NoSignal
from trader.events import closed_candles_as_of

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
MONTHS_BACK = 3
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "quant_battery"
OUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "mtf_scan_result.json"


def _load(symbol: str, tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_{tf}.csv", parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df.sort_values("timestamp").reset_index(drop=True)


def _signal_to_dict(s: MTFSignal) -> dict:
    return {
        "strategy": s.strategy, "setup": s.setup, "symbol": s.symbol, "direction": s.direction,
        "session": s.session, "entry": s.entry, "sl": s.sl, "tp1": s.tp1, "tp2": s.tp2,
        "risk_reward": s.risk_reward, "sweep_timestamp": str(s.sweep_timestamp),
        "mss_timestamp": str(s.mss_timestamp), "fvg_confirmed_at": str(s.fvg_confirmed_at),
        "generated_at": str(s.generated_at), "trace": s.trace,
    }


def main() -> None:
    config = load_config()
    all_signals: list[dict] = []
    no_signal_reasons: dict[str, dict[str, int]] = {}  # symbol -> reason -> count

    for symbol in SYMBOLS:
        t0 = time.time()
        h1_raw, m15_raw, d1_raw = _load(symbol, "H1"), _load(symbol, "M15"), _load(symbol, "D1")
        h4_raw = (
            h1_raw.set_index("timestamp").resample("4h", origin="start_day")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
        )

        window_start = m15_raw["timestamp"].iloc[-1] - pd.DateOffset(months=MONTHS_BACK)
        m15_window = m15_raw[m15_raw["timestamp"] >= window_start].reset_index(drop=True)
        print(f"{symbol}: escaneando {m15_window['timestamp'].iloc[0]} -> {m15_window['timestamp'].iloc[-1]} ({len(m15_window)} M15 bars)")

        h4_pre = PrecomputedHistory(h4_raw, "H4", config)
        h1_pre = PrecomputedHistory(h1_raw, "H1", config)
        m15_pre = PrecomputedHistory(m15_raw, "M15", config)

        reasons: dict[str, int] = {}
        n_evaluated = 0
        for ts in m15_window["timestamp"]:
            if active_entry_session(ts) is None:
                continue
            n_evaluated += 1
            h4 = h4_pre.as_of(ts)
            h1 = h1_pre.as_of(ts)
            m15 = m15_pre.as_of(ts)
            current_price = float(m15.df["close"].iloc[-1]) if len(m15.df) else float("nan")
            d1_as_of = closed_candles_as_of(d1_raw, "D1", ts)

            cont = evaluate_continuation(symbol, h4, h1, m15, d1_as_of, config, ts, current_price)
            rev = evaluate_reversal(symbol, h4, h1, m15, d1_as_of, config, ts)

            for result in (cont, rev):
                if isinstance(result, NoSignal):
                    key = f"{result.strategy}:{result.reason}"
                    reasons[key] = reasons.get(key, 0) + 1
                else:
                    for s in result:
                        all_signals.append(_signal_to_dict(s))

        no_signal_reasons[symbol] = reasons
        print(f"  {symbol}: {n_evaluated} bars en ventana de entrada evaluados, {time.time()-t0:.1f}s, "
              f"{sum(1 for s in all_signals if s['symbol']==symbol)} señales acumuladas hasta ahora")

    # Las funciones evaluate_* responden "¿hay un setup válido en este preciso
    # momento?" (apropiado para un chequeo en vivo tipo el piloto RSI) -- no
    # "¿qué apareció NUEVO desde el último chequeo?". Un mismo sweep+MSS+FVG
    # sigue siendo válido en CADA vela M15 subsiguiente hasta que expira o se
    # opera, así que escanear vela por vela redetecta el MISMO setup una y
    # otra vez. Deduplicar por la identidad real del setup (símbolo + señal +
    # sweep/MSS/FVG) antes de reportar frecuencia -- sin esto el conteo de
    # "operaciones/semana" sería varias veces mayor al real.
    dedup_keys = set()
    unique_signals = []
    for s in all_signals:
        key = (s["symbol"], s["strategy"], s["setup"], s["direction"], s["sweep_timestamp"], s["mss_timestamp"], s["fvg_confirmed_at"])
        if key not in dedup_keys:
            dedup_keys.add(key)
            unique_signals.append(s)

    with open(OUT_PATH, "w") as f:
        json.dump({
            "signals_raw_per_bar_check": all_signals, "signals_unique": unique_signals,
            "no_signal_reasons": no_signal_reasons, "months_back": MONTHS_BACK, "symbols": SYMBOLS,
        }, f, indent=2, default=str)
    print(f"\nGuardado en {OUT_PATH}: {len(all_signals)} señales crudas (por chequeo de vela) -> {len(unique_signals)} setups únicos tras deduplicar")


if __name__ == "__main__":
    main()
