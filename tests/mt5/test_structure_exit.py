import pandas as pd

from tradingai.core.signal import Direction
from tradingai.mt5.structure_exit import compute_dynamic_take_profit, structure_invalidated


def _candles(lows: list[float], highs: list[float] | None = None) -> pd.DataFrame:
    highs = highs if highs is not None else [low + 0.0010 for low in lows]
    return pd.DataFrame({"low": lows, "high": highs})


def test_long_invalidated_by_a_lower_low_break_of_structure():
    # Dos swing lows confirmados: 1.0995 (mas viejo) y 1.0980 (mas reciente, mas
    # bajo) -- un minimo mas bajo estando en largo es una ruptura de estructura.
    lows = [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1030, 1.0980, 1.0990, 1.1000, 1.1010]
    assert structure_invalidated(_candles(lows), Direction.LONG) is True


def test_long_not_invalidated_by_a_higher_low():
    # El swing mas reciente (1.1005) es MAS ALTO que el anterior (1.0995) -- minimos
    # ascendentes, estructura de tendencia alcista intacta.
    lows = [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1030, 1.1005, 1.1015, 1.1025, 1.1035]
    assert structure_invalidated(_candles(lows), Direction.LONG) is False


def test_long_not_invalidated_when_not_enough_swings_confirmed_yet():
    # Solo hay UN swing low confirmado -- no hay con que comparar, no se asume ruptura.
    lows = [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025]
    assert structure_invalidated(_candles(lows), Direction.LONG) is False


def test_short_invalidated_by_a_higher_high_break_of_structure():
    highs = [1.0970, 1.0980, 1.0990, 1.1005, 1.0995, 1.0985, 1.0975, 1.0965, 1.0955, 1.0970, 1.1020, 1.1010, 1.1000, 1.0990]
    assert structure_invalidated(_candles(lows=[h - 0.0010 for h in highs], highs=highs), Direction.SHORT) is True


def test_short_not_invalidated_by_a_lower_high():
    # El swing mas reciente (1.0995) es MAS BAJO que el anterior (1.1005) -- maximos
    # descendentes, estructura de tendencia bajista intacta.
    highs = [1.0970, 1.0980, 1.0990, 1.1005, 1.0995, 1.0985, 1.0975, 1.0965, 1.0955, 1.0970, 1.0995, 1.0985, 1.0975, 1.0965]
    assert structure_invalidated(_candles(lows=[h - 0.0010 for h in highs], highs=highs), Direction.SHORT) is False


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
