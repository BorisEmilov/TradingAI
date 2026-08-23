"""Carga de datos OHLCV desde CSV (historico) o desde el conector MT5 (en vivo)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Carga un CSV de velas y normaliza nombres/tipos de columnas."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[OHLCV_COLUMNS]


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
