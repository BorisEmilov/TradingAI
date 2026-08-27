import pytest

from tradingai.core.instruments import compute_sl_tp_with_floor, min_sl_distance_price, pip_size


def test_pip_size_forex_default():
    assert pip_size("EURUSD") == 0.0001


def test_pip_size_jpy_pairs():
    assert pip_size("USDJPY") == 0.01


def test_pip_size_index():
    assert pip_size("US500") == 0.01


def test_min_sl_distance_price_forex():
    assert min_sl_distance_price("EURUSD", min_pips=12.0) == pytest.approx(0.0012)


def test_min_sl_distance_price_jpy():
    assert min_sl_distance_price("USDJPY", min_pips=12.0) == pytest.approx(0.12)


def test_min_sl_distance_price_index_disabled():
    # "pips" de forex no aplica a un indice -- sin piso.
    assert min_sl_distance_price("US500", min_pips=12.0) == 0.0


def test_compute_sl_tp_leaves_wide_enough_sl_untouched():
    # ATR "normal": sl_atr_mult*atr ya supera el piso -> no deberia tocarse nada.
    sl, tp = compute_sl_tp_with_floor(
        entry_price=1.1000, sign=1, atr=0.0020, sl_atr_mult=2.0, tp_atr_mult=4.0,
        symbol="EURUSD", min_sl_pips=12.0,
    )
    assert sl == pytest.approx(1.1000 - 0.0040)
    assert tp == pytest.approx(1.1000 + 0.0080)


def test_compute_sl_tp_widens_sl_when_atr_too_low():
    # ATR muy bajo: sl_atr_mult*atr = 2*0.0003 = 0.0006 (6 pips) < piso de 12 pips (0.0012).
    sl, tp = compute_sl_tp_with_floor(
        entry_price=1.1000, sign=1, atr=0.0003, sl_atr_mult=2.0, tp_atr_mult=4.0,
        symbol="EURUSD", min_sl_pips=12.0,
    )
    assert sl == pytest.approx(1.1000 - 0.0012)
    # TP escalado por el mismo ratio configurado (tp_atr_mult/sl_atr_mult=2), no por
    # la distancia original -- risk:reward se mantiene igual (2:1).
    assert tp == pytest.approx(1.1000 + 0.0024)
    risk = abs(1.1000 - sl)
    reward = abs(tp - 1.1000)
    assert reward / risk == pytest.approx(2.0)


def test_compute_sl_tp_widening_preserves_configured_risk_reward_for_short():
    sl, tp = compute_sl_tp_with_floor(
        entry_price=1.1000, sign=-1, atr=0.0002, sl_atr_mult=2.0, tp_atr_mult=4.0,
        symbol="GBPJPY", min_sl_pips=12.0,
    )
    # GBPJPY es JPY -> pip=0.01, piso=12*0.01=0.12.
    assert sl == pytest.approx(1.1000 + 0.12)
    assert tp == pytest.approx(1.1000 - 0.24)


def test_compute_sl_tp_does_not_apply_floor_to_index():
    # US500: sin piso, el SL/TP sale directo del ATR aunque sea muy corto.
    sl, tp = compute_sl_tp_with_floor(
        entry_price=7680.0, sign=1, atr=0.5, sl_atr_mult=2.0, tp_atr_mult=4.0,
        symbol="US500", min_sl_pips=12.0,
    )
    assert sl == pytest.approx(7680.0 - 1.0)
    assert tp == pytest.approx(7680.0 + 2.0)
