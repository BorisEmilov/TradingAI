import numpy as np
import pandas as pd

from tradingai.core.signal import Direction
from tradingai.mt5.structure_exit import compute_dynamic_take_profit, structure_invalidated


def _trending_candles(direction: str, n: int = 300) -> pd.DataFrame:
    """Serie de velas con tendencia clara y suficiente historia para que EMA20/50/200
    queden alineadas en el sesgo correspondiente (ver indicators.add_indicators)."""
    step = 0.0005 if direction == "up" else -0.0005
    close = 1.1000 + step * np.arange(n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
        }
    )


def _candles(lows: list[float], highs: list[float] | None = None) -> pd.DataFrame:
    highs = highs if highs is not None else [low + 0.0010 for low in lows]
    return pd.DataFrame({"low": lows, "high": highs})


def test_long_invalidated_when_bias_turns_bearish():
    candles = _trending_candles("down")
    assert structure_invalidated(candles, Direction.LONG) is True


def test_long_not_invalidated_while_bias_stays_bullish():
    candles = _trending_candles("up")
    assert structure_invalidated(candles, Direction.LONG) is False


def test_short_invalidated_when_bias_turns_bullish():
    candles = _trending_candles("up")
    assert structure_invalidated(candles, Direction.SHORT) is True


def test_short_not_invalidated_while_bias_stays_bearish():
    candles = _trending_candles("down")
    assert structure_invalidated(candles, Direction.SHORT) is False


def test_long_extends_tp_to_swing_beyond_current_target():
    # Swing high confirmado en 1.1050, mas alla del TP actual (1.1030).
    highs = [1.1010, 1.1020, 1.1030, 1.1050, 1.1040, 1.1030, 1.1020, 1.1010]
    candles = _candles(lows=[h - 0.0010 for h in highs], highs=highs)
    new_tp = compute_dynamic_take_profit(
        candles, Direction.LONG, current_tp=1.1030, current_price=1.1015,
    )
    assert new_tp == 1.1050


def test_long_ignores_swing_that_does_not_extend_tp():
    # Swing high confirmado (1.1030) no supera el TP actual (1.1030) -> no se toca.
    highs = [1.1010, 1.1020, 1.1030, 1.1050, 1.1040, 1.1030, 1.1020, 1.1010]
    candles = _candles(lows=[h - 0.0010 for h in highs], highs=highs)
    new_tp = compute_dynamic_take_profit(
        candles, Direction.LONG, current_tp=1.1050, current_price=1.1015,
    )
    assert new_tp is None


def test_short_extends_tp_to_swing_beyond_current_target():
    lows = [1.0990, 1.0980, 1.0970, 1.0950, 1.0960, 1.0970, 1.0980, 1.0990]
    candles = _candles(lows)
    new_tp = compute_dynamic_take_profit(
        candles, Direction.SHORT, current_tp=1.0970, current_price=1.0985,
    )
    assert new_tp == 1.0950


def test_ignores_swing_that_would_leave_no_room_to_current_price():
    highs = [1.1010, 1.1020, 1.1030, 1.1050, 1.1040, 1.1030, 1.1020, 1.1010]
    candles = _candles(lows=[h - 0.0010 for h in highs], highs=highs)
    new_tp = compute_dynamic_take_profit(
        candles, Direction.LONG, current_tp=1.1030, current_price=1.1060,
    )
    assert new_tp is None
