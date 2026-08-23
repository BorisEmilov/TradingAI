import torch

from tradingai.ai.models.base import MultiTimeframeTradingModel
from tradingai.utils.seed import set_seed


def _model_weights_sum(seed: int) -> float:
    set_seed(seed)
    model = MultiTimeframeTradingModel(n_features=10, hidden_size=16, fusion_hidden_size=32, num_layers=1, num_heads=2)
    return sum(p.sum().item() for p in model.parameters())


def test_same_seed_gives_identical_init():
    assert _model_weights_sum(42) == _model_weights_sum(42)


def test_different_seed_gives_different_init():
    assert _model_weights_sum(1) != _model_weights_sum(2)


def test_shuffle_order_is_reproducible():
    set_seed(42)
    a = torch.randperm(20).tolist()
    set_seed(42)
    b = torch.randperm(20).tolist()
    assert a == b
