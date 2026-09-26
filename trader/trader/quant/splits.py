"""prompt-fase-cuantitativa.md: chronological EXPLORATION/VALIDATION/TEST split,
loaded from the local cache written by scripts/fetch_quant_battery_data.py --
never hits the gateway.

D1 gets a hard floor at 1999-01-01, applied to ALL 3 symbols uniformly, not just
EURUSD: the raw D1 history goes back to 1971 for EURUSD/USDJPY and 1993 for
GBPUSD, but EURUSD could not have traded before the Euro's actual introduction
(1999-01-01) -- spot-checked in the cached CSV, EURUSD already shows
EURUSD-plausible prices (~1.16-1.19) a full month before that date with a flat
spread=50 and low round-ish volume, the signature of a broker-backfilled
synthetic continuation series, not real trading. Using it would be fabricated
data. GBPUSD/USDJPY prices right at 1999-01-04 are historically plausible
(GBPUSD~1.66, USDJPY~112) with realistic-looking volume, so 1999-01-01 is a
correctness floor for EURUSD and a (generous) comparability floor for the
other two -- applying it uniformly keeps every cross-symbol comparison in the
same regime window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "quant_battery"
D1_FLOOR = pd.Timestamp("1999-01-01", tz="UTC")
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # exploration, validation, test


@dataclass(frozen=True)
class Splits:
    symbol: str
    timeframe: str
    exploration: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    if timeframe == "D1":
        df = df[df["timestamp"] >= D1_FLOOR].reset_index(drop=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, fractions: tuple[float, float, float] = SPLIT_FRACTIONS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(fractions) - 1.0) < 1e-9
    n = len(df)
    n_explore = int(n * fractions[0])
    n_validation = int(n * fractions[1])
    explore = df.iloc[:n_explore].reset_index(drop=True)
    validation = df.iloc[n_explore : n_explore + n_validation].reset_index(drop=True)
    test = df.iloc[n_explore + n_validation :].reset_index(drop=True)
    return explore, validation, test


def load_splits(symbol: str, timeframe: str) -> Splits:
    df = load_candles(symbol, timeframe)
    explore, validation, test = chronological_split(df)
    return Splits(symbol=symbol, timeframe=timeframe, exploration=explore, validation=validation, test=test)
