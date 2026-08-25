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
from threadpoolctl import threadpool_limits

from tradingai.ai.data.features.pipeline import build_feature_pipeline
from tradingai.ai.data.multi_timeframe import TIMEFRAMES
from tradingai.ai.data.preprocessor import normalize_ohlcv
from tradingai.core.signal import Direction, TradingSignal

_DIRECTION_MAP = {0: Direction.NEUTRAL, 1: Direction.LONG, 2: Direction.SHORT}

# Fallback si el checkpoint es de antes de que estos multiplicadores fueran
# configurables (ver config.yaml: model.tp_atr_mult/sl_atr_mult) -- coincide con el
# default historico de triple_barrier_labels().
_DEFAULT_TP_ATR_MULT = 2.0
_DEFAULT_SL_ATR_MULT = 1.0


class GBMPredictor:
    def __init__(self, models: list, config: dict, feature_columns: list[str]) -> None:
        # Ensemble de N modelos (semillas distintas, mismos datos/arquitectura):
        # predict() promedia sus probabilidades para reducir la varianza de "le toco
        # una inicializacion buena/mala" (ver scripts/train_gbm.py). Un checkpoint
        # con un solo modelo sigue funcionando igual (ensemble de tamano 1).
        self.models = models
        self.config = config
        self.feature_columns = feature_columns
        # No hace falta la secuencia completa para predecir, pero se mantiene para que
        # el buffer de velas a pedir (seq_len + warm-up) sea igual que con el transformer.
        self.seq_len_by_tf: dict[str, int] = config["model"]["sequence_length"]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, feature_columns: list[str] | None = None) -> "GBMPredictor":
        checkpoint = joblib.load(checkpoint_path)
        columns = feature_columns or checkpoint["feature_columns"]
        # Compatibilidad con checkpoints de antes del ensemble (guardaban "model" en
        # singular, un solo modelo).
        models = checkpoint["models"] if "models" in checkpoint else [checkpoint["model"]]
        return cls(models, checkpoint["config"], columns)

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
        # Sin este limite, cada llamada a predict_proba abre su propio pool de hilos
        # BLAS/OpenMP (~10 hilos x 5 modelos del ensemble). Con 16 simbolos corriendo
        # en paralelo y sincronizados en el cierre de vela M15, eso satura la maquina
        # (load average >200, ver memoria feedback-hardware-thermal-limits) para una
        # inferencia que de por si es casi instantanea con 1 solo hilo.
        with threadpool_limits(limits=1):
            probs = np.mean([model.predict_proba(x)[0] for model in self.models], axis=0)
        direction_idx = int(np.argmax(probs))
        confidence = float(probs[direction_idx])
        direction = _DIRECTION_MAP[direction_idx]

        last_close = float(candles_by_tf["M15"]["close"].iloc[-1])
        atr = float(m15_features["atr_14"].iloc[-1]) if "atr_14" in m15_features.columns else None

        model_cfg = self.config.get("model", {})
        tp_atr_mult = model_cfg.get("tp_atr_mult", _DEFAULT_TP_ATR_MULT)
        sl_atr_mult = model_cfg.get("sl_atr_mult", _DEFAULT_SL_ATR_MULT)

        take_profit = stop_loss = None
        if atr is not None and direction != Direction.NEUTRAL:
            sign = 1 if direction == Direction.LONG else -1
            take_profit = last_close + sign * tp_atr_mult * atr
            stop_loss = last_close - sign * sl_atr_mult * atr

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
