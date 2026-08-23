"""Normalizacion y construccion de secuencias (ventanas) para el modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza precios como retornos relativos a la vela anterior (evita look-ahead bias)."""
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        out[f"{col}_ret"] = out[col].pct_change()
    out["volume_z"] = (out["volume"] - out["volume"].rolling(50).mean()) / out["volume"].rolling(50).std()

    # Solo se exige ausencia de NaN en columnas numericas: columnas como
    # "structure_event" usan None legitimamente para "sin evento en esta vela"
    # y no deben tirar filas por warm-up de indicadores.
    numeric_cols = out.select_dtypes(include=["number", "bool"]).columns
    return out.dropna(subset=numeric_cols).reset_index(drop=True)


def build_sequences(
    features: pd.DataFrame,
    sequence_length: int,
    feature_columns: list[str],
) -> np.ndarray:
    """Convierte un DataFrame de features en ventanas deslizantes (N, sequence_length, n_features)."""
    values = features[feature_columns].to_numpy(dtype=np.float32)
    n = len(values) - sequence_length + 1
    if n <= 0:
        raise ValueError("No hay suficientes filas para construir ni una secuencia.")
    return np.stack([values[i : i + sequence_length] for i in range(n)])
