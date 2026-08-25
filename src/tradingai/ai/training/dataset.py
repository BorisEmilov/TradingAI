"""Dataset supervisado multi-temporalidad: secuencias D1/H1/M15/M5 + etiquetas.

Cada ejemplo se ancla en el cierre de una vela M15 (ver ai.data.multi_timeframe).
Etiquetado por "triple barrera" simplificado sobre esa vela: se mira hacia adelante
`horizon` velas M15 y se comprueba si el precio toca antes una barrera superior o
inferior (multiplos de ATR). Eso define direction/tp/sl objetivo. Es un metodo de
partida razonable; se puede sustituir por otro (p.ej. basado en el proximo swing de
estructura) sin tocar el resto del pipeline.

`tp_atr_mult`/`sl_atr_mult` por defecto dan un ratio 2:1, igual al piso duro que aplica
`RiskManager` en ejecucion (ver MIN_RISK_REWARD_RATIO) — el etiquetado y la aprobacion
de senales en vivo nunca deberian estar en desacuerdo sobre cual es el R:R minimo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tradingai.ai.data.multi_timeframe import TIMEFRAMES, align_and_build_sequences
from tradingai.core.signal import MIN_RISK_REWARD_RATIO


def triple_barrier_labels(
    df,
    horizon: int = 20,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
) -> dict[str, np.ndarray]:
    if tp_atr_mult / sl_atr_mult < MIN_RISK_REWARD_RATIO:
        raise ValueError(
            f"tp_atr_mult/sl_atr_mult ({tp_atr_mult / sl_atr_mult:.2f}) esta por debajo del "
            f"piso de risk:reward ({MIN_RISK_REWARD_RATIO}); el etiquetado no puede contradecir "
            f"la regla que aplica RiskManager en ejecucion."
        )

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr_14"].to_numpy() if "atr_14" in df.columns else np.full(len(df), np.nan)

    n = len(df)
    direction = np.zeros(n, dtype=np.int64)  # 0=neutral,1=long,2=short (ver Direction)
    entry_offset = np.zeros(n, dtype=np.float32)
    tp_mult = np.full(n, tp_atr_mult, dtype=np.float32)
    sl_mult = np.full(n, sl_atr_mult, dtype=np.float32)

    # Vela puntual para entrada fina en M5: su propio rango ya se movio mucho respecto
    # a su ATR, así que esperar confirmacion en M15 daria un precio de entrada peor.
    # Heuristico simple y explicable, no una regla de trading validada; punto natural
    # para reemplazar por algo mas sofisticado mas adelante.
    candle_range = high - low
    entry_timeframe = np.zeros(n, dtype=np.int64)  # 0=M15, 1=M5 (ver EntryTimeframeHead)
    with np.errstate(divide="ignore", invalid="ignore"):
        range_over_atr = np.where(atr > 0, candle_range / atr, 0.0)
    entry_timeframe[range_over_atr > 1.5] = 1

    for i in range(n - horizon):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        entry = close[i]
        upper = entry + tp_atr_mult * atr[i]
        lower = entry - sl_atr_mult * atr[i]

        future_high = high[i + 1 : i + 1 + horizon]
        future_low = low[i + 1 : i + 1 + horizon]

        hit_upper = np.argmax(future_high >= upper) if (future_high >= upper).any() else None
        hit_lower = np.argmax(future_low <= lower) if (future_low <= lower).any() else None

        if hit_upper is not None and (hit_lower is None or hit_upper <= hit_lower):
            direction[i] = 1  # long: toca TP alcista antes que el SL
        elif hit_lower is not None:
            direction[i] = 2  # short: toca la barrera bajista antes
        else:
            direction[i] = 0  # neutral: no toca ninguna barrera en el horizonte

    return {
        "direction": direction,
        "entry_offset": entry_offset,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "entry_timeframe": entry_timeframe,
    }


class MultiTimeframeTradingDataset(Dataset):
    def __init__(
        self,
        features_by_tf: dict[str, pd.DataFrame],
        feature_columns: list[str],
        seq_len_by_tf: dict[str, int],
        horizon: int = 20,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.0,
    ) -> None:
        # Guardado para que la validacion (walk-forward) pueda purgar correctamente
        # los ejemplos cuya etiqueta mira hacia adelante mas alla del corte train/test
        # (ver scripts/baseline_gbm.py::_purge_train_end) sin duplicar el numero aqui.
        self.horizon = horizon

        aligned = align_and_build_sequences(features_by_tf, feature_columns, seq_len_by_tf)
        self.sequences = aligned.sequences  # {tf: (N, seq_len, n_features)}
        # Indice (en features_by_tf["M15"]) de cada ejemplo, en el mismo orden que el
        # dataset. Util para saber, fuera de aqui, que rango de fechas cubre cada split
        # (p.ej. para evaluar un backtest solo sobre el tramo de validacion).
        self.anchor_positions = aligned.anchor_positions

        labels = triple_barrier_labels(
            features_by_tf["M15"], horizon=horizon, tp_atr_mult=tp_atr_mult, sl_atr_mult=sl_atr_mult
        )
        idx = aligned.anchor_positions
        self.direction = labels["direction"][idx]
        self.entry_offset = labels["entry_offset"][idx]
        self.tp_mult = labels["tp_mult"][idx]
        self.sl_mult = labels["sl_mult"][idx]
        self.entry_timeframe = labels["entry_timeframe"][idx]

    def __len__(self) -> int:
        return len(self.direction)

    def __getitem__(self, idx: int):
        item = {tf: torch.from_numpy(self.sequences[tf][idx]) for tf in TIMEFRAMES}
        item.update(
            direction=torch.tensor(self.direction[idx], dtype=torch.long),
            entry_offset=torch.tensor(self.entry_offset[idx], dtype=torch.float32),
            tp_mult=torch.tensor(self.tp_mult[idx], dtype=torch.float32),
            sl_mult=torch.tensor(self.sl_mult[idx], dtype=torch.float32),
            entry_timeframe=torch.tensor(self.entry_timeframe[idx], dtype=torch.long),
        )
        return item
