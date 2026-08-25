import pandas as pd

from tradingai.core.signal import Direction
from tradingai.mt5.trailing_stop import compute_trailing_sl


def _candles(lows: list[float], highs: list[float] | None = None) -> pd.DataFrame:
    highs = highs if highs is not None else [low + 0.0010 for low in lows]
    return pd.DataFrame({"low": lows, "high": highs})


def test_long_does_not_activate_before_reaching_r_multiple():
    # Riesgo inicial = 1.1000 - 1.0980 = 0.0020. Precio actual solo se movio 0.0010
    # a favor (0.5R) -- no deberia activarse todavia con el default de 1R.
    candles = _candles([1.0975, 1.0985, 1.0995, 1.1005, 1.1015, 1.1010, 1.1008, 1.1009])
    new_sl = compute_trailing_sl(
        candles, Direction.LONG, entry_price=1.1000, current_sl=1.0980, current_price=1.1010,
    )
    assert new_sl is None


def test_long_moves_sl_to_last_confirmed_swing_low_once_activated():
    # Riesgo inicial = 0.0020. Precio se movio 0.0030 a favor (1.5R) -> activado.
    # El swing low confirmado (minimo local con 3 velas a cada lado) esta en 1.0995.
    lows = [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035]
    candles = _candles(lows)
    new_sl = compute_trailing_sl(
        candles, Direction.LONG, entry_price=1.1000, current_sl=1.0980, current_price=1.1030,
    )
    assert new_sl == 1.0995


def test_long_never_loosens_the_stop():
    # El swing detectado (1.0995) es PEOR que el SL actual (1.0999) -> no se mueve.
    lows = [1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035]
    candles = _candles(lows)
    new_sl = compute_trailing_sl(
        candles, Direction.LONG, entry_price=1.1000, current_sl=1.0999, current_price=1.1030,
    )
    assert new_sl is None


def test_short_moves_sl_to_last_confirmed_swing_high_once_activated():
    # Simetrico al caso LONG: el swing high confirmado esta en 1.1005.
    highs = [1.0970, 1.0980, 1.0990, 1.1005, 1.0995, 1.0985, 1.0975, 1.0965]
    candles = _candles(lows=[h - 0.0010 for h in highs], highs=highs)
    new_sl = compute_trailing_sl(
        candles, Direction.SHORT, entry_price=1.1000, current_sl=1.1020, current_price=1.0970,
    )
    assert new_sl == 1.1005


def test_returns_none_when_no_swing_confirmed_yet():
    # Serie estrictamente descendente: nunca hay un minimo local confirmado.
    lows = [1.1000, 1.0990, 1.0980, 1.0970, 1.0960, 1.0950, 1.0940]
    candles = _candles(lows)
    new_sl = compute_trailing_sl(
        candles, Direction.LONG, entry_price=1.1000, current_sl=1.0950, current_price=1.1200,
    )
    assert new_sl is None


def test_ignores_swing_that_would_leave_no_room_to_current_price():
    # El swing confirmado (1.1005) queda POR ENCIMA del precio actual (1.1000) -> se
    # ignora en vez de arriesgar un cierre inmediato, aunque el trailing ya este
    # activado (2.5R de beneficio).
    lows = [1.1030, 1.1020, 1.1010, 1.1005, 1.1015, 1.1020, 1.1025, 1.1030]
    candles = _candles(lows)
    new_sl = compute_trailing_sl(
        candles, Direction.LONG, entry_price=1.0950, current_sl=1.0930, current_price=1.1000,
    )
    assert new_sl is None
