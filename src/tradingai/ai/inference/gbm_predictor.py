"""Predictor de linea base: gradient boosting sobre el snapshot mas reciente de cada
temporalidad, sin secuencias completas.

En el walk-forward del 2026-08-23 (ver memoria del proyecto) este enfoque simple
supero de forma consistente al transformer multi-temporalidad: 20/20 folds positivos
en los 5 simbolos probados, contra un transformer inestable que oscilaba entre 5% y
80% de acierto segun el fold. Por eso es el modelo real, no el transformer.

Solo predice DIRECCION. El TP/SL se construye con los mismos multiplicadores ATR fijos
que usa el etiquetado por triple-barrera (ver ai.training.dataset.triple_barrier_labels)
— este modelo no aprende un TP/SL variable como intentaba hacer el transformer.

Expone la misma interfaz que `Predictor` (`.seq_len_by_tf`, `.predict(candles_by_tf,
symbol) -> TradingSignal`) para poder usarse en `TradingPipeline`/`backtest.py`/
`run_live.py` sin cambios en esos modulos.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from tradingai.ai.data.features.pipeline import build_feature_pipeline
from tradingai.ai.data.multi_timeframe import TIMEFRAMES
from tradingai.ai.data.preprocessor import normalize_ohlcv
from tradingai.core.signal import Direction, TradingSignal

_DIRECTION_MAP = {0: Direction.NEUTRAL, 1: Direction.LONG, 2: Direction.SHORT}

# Mismos multiplicadores fijos que el etiquetado (ai.training.dataset.triple_barrier_labels).
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0


class GBMPredictor:
    def __init__(self, model, config: dict, feature_columns: list[str]) -> None:
        self.model = model
        self.config = config
        self.feature_columns = feature_columns
        # No hace falta la secuencia completa para predecir, pero se mantiene para que
        # el buffer de velas a pedir (seq_len + warm-up) sea igual que con el transformer.
        self.seq_len_by_tf: dict[str, int] = config["model"]["sequence_length"]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, feature_columns: list[str] | None = None) -> "GBMPredictor":
        checkpoint = joblib.load(checkpoint_path)
        columns = feature_columns or checkpoint["feature_columns"]
        return cls(checkpoint["model"], checkpoint["config"], columns)

    def predict(self, candles_by_tf: dict, symbol: str) -> TradingSignal:
        missing = [tf for tf in TIMEFRAMES if tf not in candles_by_tf]
        if missing:
            raise ValueError(f"Faltan velas de estas temporalidades: {missing}")

        last_rows = []
        m15_features = None
        for tf in TIMEFRAMES:
            features = normalize_ohlcv(build_feature_pipeline(candles_by_tf[tf], self.config))
            if len(features) == 0:
                raise ValueError(f"No hay velas con features validas en {tf}")
            last_rows.append(features[self.feature_columns].iloc[-1].to_numpy(dtype="float32"))
            if tf == "M15":
                m15_features = features

        x = np.concatenate(last_rows).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        direction_idx = int(np.argmax(probs))
        confidence = float(probs[direction_idx])
        direction = _DIRECTION_MAP[direction_idx]

        last_close = float(candles_by_tf["M15"]["close"].iloc[-1])
        atr = float(m15_features["atr_14"].iloc[-1]) if "atr_14" in m15_features.columns else None

        take_profit = stop_loss = None
        if atr is not None and direction != Direction.NEUTRAL:
            sign = 1 if direction == Direction.LONG else -1
            take_profit = last_close + sign * TP_ATR_MULT * atr
            stop_loss = last_close - sign * SL_ATR_MULT * atr

        return TradingSignal(
            symbol=symbol,
            timeframe="M15",
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            confidence=confidence,
            entry_price=last_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_timeframe="M15",
            rationale={"direction_probs": probs.tolist()},
        )
