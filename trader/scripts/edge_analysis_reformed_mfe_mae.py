"""Re-ejecucion de prompt-analisis-edge-real.md (Partes 1-2) sobre el
sistema reformulado de 4 capas: mide si el momento de entrada que produce la
Capa 3 (confirmacion de price action amplia, trader/confirmation.py) ubica
momentos donde el precio realmente tiende a moverse a favor -- MFE/MAE SIN
aplicar SL/TP/parcial/gestion -- comparado contra un control aleatorio con
las mismas restricciones de sesion/noticias.

Deliberadamente NO exige razon dominante (Capa 2) ni gates de riesgo/R:R
(Capa 4): igual que el script original solo exigia POI+confirmacion (no
score/risk/R:R, que son sobre la CALIDAD del trade, no sobre si el momento
de entrada tiene edge), este mide la Capa 3 sola -- el reemplazo directo del
"catalogo cerrado de 4 patrones" que la version anterior probaba.

MFE/MAE normalizado por ATR(M15) en el momento de la senal, mismo criterio
que el script original, para que el grupo aleatorio tenga una unidad de
riesgo comparable.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from trader.config import load_config
from trader.confirmation import compute_confirmations
from trader.events import TF_DURATION
from trader.gateway_client import PythonGetawayClient
from trader.news_filter import is_high_impact_news_window
from trader.pipeline.engine import TimeframeAnalysis
from trader.sessions import classify_session

M15_COUNT = 50000
WINDOW_BARS = 96  # 24h on M15 -- same fixed horizon as the original script
SEED = 20260918  # same seed as the original run, documented not tuned after seeing results


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


def reformed_candidates(symbol: str, m15_df: pd.DataFrame, m15_analysis: TimeframeAnalysis, config) -> list[dict]:
    confirmations = compute_confirmations(m15_df)
    atr_series = m15_analysis.atr
    n = len(m15_df)
    rows = []
    for i in range(n - 1):
        row = confirmations.get(i, {})
        anchor_idx = i + 1
        as_of = m15_df["timestamp"].iloc[anchor_idx] + TF_DURATION["M15"]
        for direction in ("bullish", "bearish"):
            confirmation = row.get(direction)
            if confirmation is None:
                continue
            if not _session_news_ok(as_of, config):
                continue
            atr_at_entry = (
                float(atr_series.iloc[anchor_idx])
                if anchor_idx < len(atr_series) and not pd.isna(atr_series.iloc[anchor_idx])
                else None
            )
            if atr_at_entry is None or atr_at_entry <= 0:
                continue
            current_price = float(m15_df["close"].iloc[anchor_idx])
            mfe_r, mae_r, race = _mfe_mae(direction, current_price, m15_df, anchor_idx, atr_at_entry)
            rows.append(dict(
                symbol=symbol, as_of=as_of, direction=direction, entry=current_price, bar_idx=anchor_idx,
                mfe_r=mfe_r, mae_r=mae_r, race=race, confirmation_score_percentile=confirmation.score_percentile,
            ))
    return rows


def random_candidates(
    symbol: str, m15_df: pd.DataFrame, m15_analysis: TimeframeAnalysis, config, n: int, rng: random.Random
) -> list[dict]:
    m15_close_ts = (m15_df["timestamp"] + TF_DURATION["M15"]).tolist()
    atr_series = m15_analysis.atr

    eligible_idx = []
    for i, as_of in enumerate(m15_close_ts):
        if i >= len(atr_series) or pd.isna(atr_series.iloc[i]) or atr_series.iloc[i] <= 0:
            continue
        if not _session_news_ok(as_of, config):
            continue
        eligible_idx.append(i)

    chosen = rng.sample(eligible_idx, min(n, len(eligible_idx)))
    rows = []
    for bar_idx in chosen:
        as_of = m15_close_ts[bar_idx]
        current_price = float(m15_df["close"].iloc[bar_idx])
        direction = "bullish" if rng.random() < 0.5 else "bearish"
        atr_at_entry = float(atr_series.iloc[bar_idx])
        mfe_r, mae_r, race = _mfe_mae(direction, current_price, m15_df, bar_idx, atr_at_entry)
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

    all_reformed = []
    all_random = []
    rng = random.Random(SEED)

    try:
        for symbol in config.symbols:
            print(f"=== {symbol}: fetching + detectores ===")
            m15_df = client.candles(symbol, "M15", M15_COUNT, timeout=120)
            m15_analysis = TimeframeAnalysis.from_candles(m15_df, "M15", config)

            reformed_rows = reformed_candidates(symbol, m15_df, m15_analysis, config)
            print(f"  Capa 3 candidatos (confirmacion amplia valida): {len(reformed_rows)}")
            random_rows = random_candidates(symbol, m15_df, m15_analysis, config, len(reformed_rows), rng)
            print(f"  Random candidatos (control): {len(random_rows)}")

            all_reformed.extend(reformed_rows)
            all_random.extend(random_rows)
    finally:
        client.logout()

    print("\n=== RESUMEN AGREGADO (3 simbolos) ===\n")
    _summarize(all_reformed, "Senales Capa 3 (confirmacion de price action amplia)")
    print()
    _summarize(all_random, "Control aleatorio (mismo n, mismas restricciones de sesion/noticias)")

    logs_dir = Path(config.signals_log_path).parent
    with (logs_dir / "edge_analysis_reformed_candidates.jsonl").open("w") as fh:
        for r in all_reformed:
            fh.write(json.dumps({**r, "as_of": r["as_of"].isoformat()}) + "\n")
    with (logs_dir / "edge_analysis_reformed_random_candidates.jsonl").open("w") as fh:
        for r in all_random:
            fh.write(json.dumps({**r, "as_of": r["as_of"].isoformat()}) + "\n")
    print(f"\nGuardado: {logs_dir}/edge_analysis_reformed_candidates.jsonl, edge_analysis_reformed_random_candidates.jsonl")


if __name__ == "__main__":
    main()
