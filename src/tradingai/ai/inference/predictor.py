"""Convierte velas OHLCV multi-temporalidad en un TradingSignal: features -> modelo -> senal.

Requiere velas de las 4 temporalidades (D1/H1/M15/M5) para la misma vela M15 actual —
el modelo replica el analisis top-down (bias D1 -> estructura H1 -> entrada M15/M5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from tradingai.ai.data.features.pipeline import build_feature_pipeline
from tradingai.ai.data.multi_timeframe import TIMEFRAMES
from tradingai.ai.data.preprocessor import normalize_ohlcv
from tradingai.ai.models.base import MultiTimeframeTradingModel, build_model
from tradingai.ai.models.entry_timeframe_model import ENTRY_TIMEFRAMES
from tradingai.core.signal import Direction, TradingSignal

_DIRECTION_MAP = {0: Direction.NEUTRAL, 1: Direction.LONG, 2: Direction.SHORT}


class Predictor:
    def __init__(self, model: MultiTimeframeTradingModel, config: dict, feature_columns: list[str]) -> None:
        self.model = model
        self.model.eval()
        self.config = config
        self.feature_columns = feature_columns
        self.seq_len_by_tf: dict[str, int] = config["model"]["sequence_length"]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, feature_columns: list[str]) -> "Predictor":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        model = build_model(config, n_features=len(feature_columns))
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, config, feature_columns)

    @torch.no_grad()
    def predict(self, candles_by_tf: dict[str, pd.DataFrame], symbol: str) -> TradingSignal:
        missing = [tf for tf in TIMEFRAMES if tf not in candles_by_tf]
        if missing:
            raise ValueError(f"Faltan velas de estas temporalidades: {missing}")

        sequences = {}
        features_by_tf = {}
        for tf in TIMEFRAMES:
            features = normalize_ohlcv(build_feature_pipeline(candles_by_tf[tf], self.config))
            seq_len = self.seq_len_by_tf[tf]
            if len(features) < seq_len:
                raise ValueError(
                    f"Se necesitan al menos {seq_len} velas con features validas en {tf}, "
                    f"hay {len(features)}."
                )
            window = features.iloc[-seq_len:]
            sequences[tf] = torch.tensor(window[self.feature_columns].to_numpy(dtype="float32")).unsqueeze(0)
            features_by_tf[tf] = features

        outputs = self.model(sequences)

        probs = torch.softmax(outputs["direction_logits"], dim=-1).squeeze(0)
        direction_idx = int(torch.argmax(probs).item())
        confidence = float(probs[direction_idx].item())
        direction = _DIRECTION_MAP[direction_idx]

        entry_tf_probs = torch.softmax(outputs["entry_timeframe_logits"], dim=-1).squeeze(0)
        entry_timeframe = ENTRY_TIMEFRAMES[int(torch.argmax(entry_tf_probs).item())]

        m15_features = features_by_tf["M15"]
        last_close = float(candles_by_tf["M15"]["close"].iloc[-1])
        atr = float(m15_features["atr_14"].iloc[-1]) if "atr_14" in m15_features.columns else None

        entry_price = last_close * (1 + float(outputs["entry_offset"].item()))
        tp_mult = float(outputs["tp_mult"].item())
        sl_mult = float(outputs["sl_mult"].item())

        take_profit = stop_loss = None
        if atr is not None and direction != Direction.NEUTRAL:
            sign = 1 if direction == Direction.LONG else -1
            take_profit = entry_price + sign * tp_mult * atr
            stop_loss = entry_price - sign * sl_mult * atr

        return TradingSignal(
            symbol=symbol,
            timeframe="M15",
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_timeframe=entry_timeframe,
            rationale={"direction_probs": probs.tolist(), "entry_timeframe_probs": entry_tf_probs.tolist()},
        )
