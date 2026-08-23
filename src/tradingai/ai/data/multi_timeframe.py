"""Alineacion multi-temporalidad: D1 (bias) -> H1 (estructura) -> M15 (entrada) -> M5 (entrada fina).

Cada decision de trading se ancla en el cierre de una vela M15 (la temporalidad de
entrada principal). Para esa vela, se construye una ventana de contexto en cada
temporalidad usando solo velas ya cerradas en ese instante (sin look-ahead): las
ultimas `seq_len[D1]` velas D1, `seq_len[H1]` velas H1, etc.

Esto reproduce como opera un trader real: mira el diario para el sesgo, baja a H1
para refinar estructura, y decide la entrada en M15 (con M5 disponible para precision
fina en casos puntuales) — en vez de una prediccion ciega sobre una sola temporalidad.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TIMEFRAMES = ("D1", "H1", "M15", "M5")
ANCHOR_TIMEFRAME = "M15"


def last_closed_bar_index(bar_timestamps: np.ndarray, anchor_timestamps: np.ndarray) -> np.ndarray:
    """Para cada anchor, indice de la ultima barra con timestamp <= anchor (-1 si ninguna)."""
    return np.searchsorted(bar_timestamps, anchor_timestamps, side="right") - 1


def _sliding_windows(values: np.ndarray, seq_len: int) -> np.ndarray:
    """(n_bars, n_features) -> (n_bars - seq_len + 1, seq_len, n_features), sin copiar."""
    windows = sliding_window_view(values, seq_len, axis=0)  # (n-seq_len+1, n_features, seq_len)
    return np.moveaxis(windows, -1, 1)


@dataclass
class MultiTimeframeSequences:
    sequences: dict[str, np.ndarray]  # {tf: (N, seq_len[tf], n_features)}, N = anchors validos
    anchor_positions: np.ndarray  # indices (en el df M15 de entrada) de cada anchor valido


def align_and_build_sequences(
    features_by_tf: dict[str, pd.DataFrame],
    feature_columns: list[str],
    seq_len_by_tf: dict[str, int],
) -> MultiTimeframeSequences:
    """Construye secuencias alineadas para cada vela M15 con suficiente historia en las 4 TF.

    `features_by_tf[tf]` debe tener una columna "timestamp" (marca de cierre de vela) y
    todas las columnas en `feature_columns`, ya sin NaN (ver normalize_ohlcv).
    """
    missing_tf = [tf for tf in TIMEFRAMES if tf not in features_by_tf]
    if missing_tf:
        raise ValueError(f"Faltan temporalidades: {missing_tf}")

    anchor_df = features_by_tf[ANCHOR_TIMEFRAME]
    anchor_timestamps = anchor_df["timestamp"].to_numpy()

    windows_by_tf: dict[str, np.ndarray] = {}
    valid_window_idx_by_tf: dict[str, np.ndarray] = {}  # por cada anchor: indice de ventana o -1

    for tf in TIMEFRAMES:
        df = features_by_tf[tf]
        seq_len = seq_len_by_tf[tf]
        values = df[feature_columns].to_numpy(dtype=np.float32)
        ts = df["timestamp"].to_numpy()

        if len(values) < seq_len:
            raise ValueError(f"No hay suficientes velas en {tf} ({len(values)}) para seq_len={seq_len}")

        windows_by_tf[tf] = _sliding_windows(values, seq_len)

        # Para cada anchor M15, la ultima barra de `tf` cerrada en tf <= al anchor.
        last_bar_idx = last_closed_bar_index(ts, anchor_timestamps)
        window_idx = last_bar_idx - seq_len + 1
        valid_window_idx_by_tf[tf] = window_idx

    valid_mask = np.ones(len(anchor_timestamps), dtype=bool)
    for tf in TIMEFRAMES:
        valid_mask &= valid_window_idx_by_tf[tf] >= 0

    anchor_positions = np.nonzero(valid_mask)[0]

    sequences = {
        tf: windows_by_tf[tf][valid_window_idx_by_tf[tf][anchor_positions]] for tf in TIMEFRAMES
    }

    return MultiTimeframeSequences(sequences=sequences, anchor_positions=anchor_positions)


def flatten_last_timestep(sequences: dict[str, np.ndarray]) -> np.ndarray:
    """Concatena el ultimo timestep de cada temporalidad en un vector plano por ejemplo:
    {tf: (N, seq_len, n_features)} -> (N, len(TIMEFRAMES) * n_features).

    Lo usa el modelo de linea base (gradient boosting), que no necesita la secuencia
    completa — solo el snapshot mas reciente de D1/H1/M15/M5 — a diferencia del
    transformer, que consume la secuencia entera.
    """
    return np.concatenate([sequences[tf][:, -1, :] for tf in TIMEFRAMES], axis=1)
