"""prompt-analisis-edge-real.md, Partes 1-2: independiente del volumen de
operaciones, mide si las senales de entrada (POI+confirmacion+bias) ubican
momentos donde el precio realmente tiende a moverse a favor -- MFE/MAE SIN
aplicar SL/TP/parcial/breakeven -- comparado contra un grupo de control de
entradas aleatorias (mismas restricciones de sesion/noticias, sin exigir POI
ni confirmacion).

Pool de candidatos ICT: reusa la arquitectura D1->M15-unico
(trader/pipeline/experiment_ltf.py) por ser la de mayor cobertura de las
probadas este proyecto -- pero SOLO hasta que exista POI+confirmacion
validos (no aplica score/risk/R:R, que son sobre la CALIDAD del trade, no
sobre si el momento de entrada tiene edge). Diagnostico puro, no toca
produccion.

MFE/MAE normalizado por ATR(M15) en el momento de la senal (no por el SL
estructural del POI) para que el grupo aleatorio -- sin POI, sin stop
estructural que definir -- tenga una unidad de riesgo comparable y no
arbitraria.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from trader.config import load_config
from trader.detectors.indicators import atr as atr_fn
from trader.events import TF_DURATION, MarketEvent
from trader.gateway_client import PythonGetawayClient
from trader.news_filter import is_high_impact_news_window
from trader.pipeline.engine import _CONFIRMATION_KINDS, TimeframeAnalysis
from trader.pipeline.experiment_ltf import LTF_15M, _candidate_confirmation_timestamps, build_ltf_analysis
from trader.sessions import classify_session


def _is_confirmation(e: MarketEvent) -> bool:
    return e.kind in _CONFIRMATION_KINDS or e.kind.startswith("inverted_fair_value_gap")


def _latest_confirmation_causal(ltf_analysis: TimeframeAnalysis, direction: str, as_of: pd.Timestamp, window_candles: int) -> MarketEvent | None:
    # `MultiTimeframePipeline._latest_confirmation` measures freshness against
    # `ltf_analysis.df.iloc[-1]` -- the LAST candle of whatever df it's given,
    # correct only when called on an as_of()-truncated analysis (every
    # production caller). Here `ltf_analysis` is the FULL-HISTORY object
    # (never truncated, to avoid `.as_of()`'s cost for a plain point-event
    # lookup -- see prompt-fase7-modelo-probabilistico.md's build_ml_dataset.py
    # for the same reasoning), so freshness must be measured against `as_of`
    # directly instead. First version of this script called the production
    # method directly and got 0 candidates across the board -- exactly this bug.
    step = TF_DURATION[ltf_analysis.timeframe]
    earliest = as_of - (window_candles - 1) * step
    candidates = [
        e for e in ltf_analysis.all_events()
        if earliest <= e.timestamp <= as_of and e.direction == direction and _is_confirmation(e)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.timestamp)

D1_COUNT = 1500
M15_COUNT = 50000
WINDOW_BARS = 96  # 24h on M15 -- fixed day-trading-relevant horizon, see module docstring
SEED = 20260918  # deterministic control group, documented not tuned after seeing results


def _mfe_mae(direction: str, entry: float, df: pd.DataFrame, entry_bar_idx: int, atr_at_entry: float) -> tuple[float, float, str]:
    mfe = 0.0
    mae = 0.0
    first_mfe1r_idx = None
    first_mae1r_idx = None
    end = min(entry_bar_idx + 1 + WINDOW_BARS, len(df))
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    for k in range(entry_bar_idx + 1, end):
        if direction == "bullish":
            fav = highs[k] - entry
            adv = entry - lows[k]
        else:
            fav = entry - lows[k]
            adv = highs[k] - entry
        if fav > mfe:
            mfe = fav
        if adv > mae:
            mae = adv
        if first_mfe1r_idx is None and atr_at_entry > 0 and fav >= atr_at_entry:
            first_mfe1r_idx = k
        if first_mae1r_idx is None and atr_at_entry > 0 and adv >= atr_at_entry:
            first_mae1r_idx = k
        if first_mfe1r_idx is not None and first_mae1r_idx is not None:
            break
    mfe_r = mfe / atr_at_entry if atr_at_entry > 0 else float("nan")
    mae_r = mae / atr_at_entry if atr_at_entry > 0 else float("nan")
    if first_mfe1r_idx is not None and (first_mae1r_idx is None or first_mfe1r_idx <= first_mae1r_idx):
        race = "mfe_first"
    elif first_mae1r_idx is not None:
        race = "mae_first"
    else:
        race = "neither"
    return mfe_r, mae_r, race


def _session_news_ok(as_of: pd.Timestamp, config) -> bool:
    session = classify_session(as_of, config.sessions)
    session_ok = (
        (session.overlap_london_ny or session.killzone_london or session.killzone_new_york)
        if config.sessions.require_overlap_or_killzone
        else session.any_active
    )
    if not session_ok:
        return False
    blocked, _ = is_high_impact_news_window(as_of, config.news_filter)
    return not blocked


def ict_candidates(symbol: str, d1_analysis: TimeframeAnalysis, ltf_analysis: TimeframeAnalysis, ltf_df: pd.DataFrame, config) -> list[dict]:
    tol = config.poi.tolerance_pct / 100.0
    m15_close_ts = ltf_df["timestamp"] + TF_DURATION["M15"]
    idx_by_close_ts = {t: i for i, t in enumerate(m15_close_ts)}

    anchors = _candidate_confirmation_timestamps(ltf_analysis, LTF_15M.window_candles)
    atr_series = ltf_analysis.atr

    rows = []
    for as_of in anchors:
        bar_idx = idx_by_close_ts.get(as_of)
        if bar_idx is None:
            continue
        current_price = float(ltf_df["close"].iloc[bar_idx])

        d1_events = [e for e in d1_analysis.structure_events if e.timestamp <= as_of]
        if not d1_events:
            continue
        if not _session_news_ok(as_of, config):
            continue

        atr_at_entry = float(atr_series.iloc[bar_idx]) if bar_idx < len(atr_series) and not pd.isna(atr_series.iloc[bar_idx]) else None
        if atr_at_entry is None or atr_at_entry <= 0:
            continue

        for direction in ("bullish", "bearish"):
            poi_candidates = [
                z for z in (ltf_analysis.order_blocks + ltf_analysis.fvgs)
                if z.direction == direction and z.confirmed_at is not None and z.confirmed_at <= as_of
                and (z.broken_at is None or as_of < z.broken_at)
                and z.overlaps(current_price * (1 - tol), current_price * (1 + tol))
            ]
            if not poi_candidates:
                continue
            confirmation = _latest_confirmation_causal(ltf_analysis, direction, as_of, LTF_15M.window_candles)
            if confirmation is None:
                continue

            mfe_r, mae_r, race = _mfe_mae(direction, current_price, ltf_df, bar_idx, atr_at_entry)
            rows.append(dict(symbol=symbol, as_of=as_of, direction=direction, entry=current_price, bar_idx=bar_idx, mfe_r=mfe_r, mae_r=mae_r, race=race))

    return rows


def random_candidates(symbol: str, d1_analysis: TimeframeAnalysis, ltf_analysis: TimeframeAnalysis, ltf_df: pd.DataFrame, config, n: int, rng: random.Random) -> list[dict]:
    m15_close_ts = (ltf_df["timestamp"] + TF_DURATION["M15"]).tolist()
    atr_series = ltf_analysis.atr
    d1_ts_sorted = [e.timestamp for e in d1_analysis.structure_events]

    eligible_idx = []
    for i, as_of in enumerate(m15_close_ts):
        if i >= len(atr_series) or pd.isna(atr_series.iloc[i]) or atr_series.iloc[i] <= 0:
            continue
        if not d1_ts_sorted or d1_ts_sorted[0] > as_of:  # no D1 bias yet at all
            continue
        if not _session_news_ok(as_of, config):
            continue
        eligible_idx.append(i)

    chosen = rng.sample(eligible_idx, min(n, len(eligible_idx)))
    rows = []
    for bar_idx in chosen:
        as_of = m15_close_ts[bar_idx]
        current_price = float(ltf_df["close"].iloc[bar_idx])
        direction = "bullish" if rng.random() < 0.5 else "bearish"
        atr_at_entry = float(atr_series.iloc[bar_idx])
        mfe_r, mae_r, race = _mfe_mae(direction, current_price, ltf_df, bar_idx, atr_at_entry)
        rows.append(dict(symbol=symbol, as_of=as_of, direction=direction, entry=current_price, bar_idx=bar_idx, mfe_r=mfe_r, mae_r=mae_r, race=race))
    return rows


def _summarize(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"{label}: 0 candidatos")
        return
    mfe = np.array([r["mfe_r"] for r in rows])
    mae = np.array([r["mae_r"] for r in rows])
    races = [r["race"] for r in rows]
    n = len(rows)
    p_mfe_first = races.count("mfe_first") / n
    p_mae_first = races.count("mae_first") / n
    p_neither = races.count("neither") / n
    print(f"{label}: n={n}")
    print(f"  MFE_R: media={mfe.mean():.3f} mediana={np.median(mfe):.3f}")
    print(f"  MAE_R: media={mae.mean():.3f} mediana={np.median(mae):.3f}")
    print(f"  P(llega a 1R de MFE antes que 1R de MAE)={p_mfe_first:.1%}, P(MAE 1R primero)={p_mae_first:.1%}, P(ninguno en {WINDOW_BARS} velas)={p_neither:.1%}")


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    all_ict = []
    all_random = []
    rng = random.Random(SEED)

    try:
        for symbol in config.symbols:
            print(f"=== {symbol}: fetching + detectores ===")
            d1_df = client.candles(symbol, "D1", D1_COUNT, timeout=120)
            m15_df = client.candles(symbol, "M15", M15_COUNT, timeout=120)
            d1_analysis = TimeframeAnalysis.from_candles(d1_df, "D1", config)
            ltf_analysis = build_ltf_analysis(m15_df, LTF_15M, config)

            ict_rows = ict_candidates(symbol, d1_analysis, ltf_analysis, m15_df, config)
            print(f"  ICT candidatos (POI+confirmacion validos): {len(ict_rows)}")
            random_rows = random_candidates(symbol, d1_analysis, ltf_analysis, m15_df, config, len(ict_rows), rng)
            print(f"  Random candidatos (control): {len(random_rows)}")

            all_ict.extend(ict_rows)
            all_random.extend(random_rows)
    finally:
        client.logout()

    print("\n=== RESUMEN AGREGADO (3 simbolos) ===\n")
    _summarize(all_ict, "Senales ICT (POI+confirmacion+bias)")
    print()
    _summarize(all_random, "Control aleatorio (mismo n, mismas restricciones de sesion/noticias)")

    logs_dir = Path(config.signals_log_path).parent
    import json

    with (logs_dir / "edge_analysis_ict_candidates.jsonl").open("w") as fh:
        for r in all_ict:
            fh.write(json.dumps({**r, "as_of": r["as_of"].isoformat()}) + "\n")
    with (logs_dir / "edge_analysis_random_candidates.jsonl").open("w") as fh:
        for r in all_random:
            fh.write(json.dumps({**r, "as_of": r["as_of"].isoformat()}) + "\n")
    print(f"\nGuardado: {logs_dir}/edge_analysis_ict_candidates.jsonl, edge_analysis_random_candidates.jsonl")


if __name__ == "__main__":
    main()
