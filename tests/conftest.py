import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _make_ohlcv(n: int, freq: str, seed: int, end: str = "2024-06-01") -> pd.DataFrame:
    """OHLCV sintetico (random walk) suficiente para ejercitar el pipeline de features.

    Se genera con `end` fijo (no `start`) para que temporalidades distintas terminen en
    el mismo instante y se solapen cerca del "ahora" — igual que en produccion, donde
    D1/H1/M15/M5 se piden hasta la vela mas reciente disponible.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.0005, n)
    close = 1.1000 * np.cumprod(1 + returns)

    high = close * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0003, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(100, 1000, n).astype(float)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(end=end, periods=n, freq=freq),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def synthetic_candles() -> pd.DataFrame:
    return _make_ohlcv(n=300, freq="15min", seed=42)


@pytest.fixture
def synthetic_multi_tf_candles() -> dict[str, pd.DataFrame]:
    """Velas sinteticas por temporalidad, todas terminando en el mismo instante,
    suficientes para pasar warm-up de indicadores (EMA-200 es el mas exigente) y
    tener secuencias de sobra para alinear."""
    return {
        "D1": _make_ohlcv(n=400, freq="1D", seed=1),
        "H1": _make_ohlcv(n=400, freq="1h", seed=2),
        "M15": _make_ohlcv(n=400, freq="15min", seed=3),
        "M5": _make_ohlcv(n=400, freq="5min", seed=4),
    }
