import torch

from tradingai.ai.data.multi_timeframe import TIMEFRAMES
from tradingai.ai.models.base import MultiTimeframeTradingModel

SEQ_LEN = {"D1": 10, "H1": 12, "M15": 14, "M5": 16}


def _fake_sequences(batch: int, n_features: int) -> dict[str, torch.Tensor]:
    return {tf: torch.randn(batch, SEQ_LEN[tf], n_features) for tf in TIMEFRAMES}


def test_transformer_forward_shapes():
    model = MultiTimeframeTradingModel(
        n_features=10, architecture="transformer", hidden_size=16, fusion_hidden_size=32,
        num_layers=1, num_heads=2,
    )
    out = model(_fake_sequences(batch=4, n_features=10))

    assert out["direction_logits"].shape == (4, 3)
    assert out["entry_offset"].shape == (4,)
    assert out["tp_mult"].shape == (4,)
    assert out["sl_mult"].shape == (4,)
    assert out["entry_timeframe_logits"].shape == (4, 2)


def test_lstm_forward_shapes():
    model = MultiTimeframeTradingModel(
        n_features=10, architecture="lstm", hidden_size=16, fusion_hidden_size=32, num_layers=1,
    )
    out = model(_fake_sequences(batch=4, n_features=10))

    assert out["direction_logits"].shape == (4, 3)
    assert out["entry_timeframe_logits"].shape == (4, 2)
