from tradingai.core.signal import Direction
from tradingai.mt5.scaled_exit import compute_tp1, should_move_to_breakeven, should_take_partial_profit


def test_compute_tp1_is_halfway_by_default_long():
    assert compute_tp1(entry_price=1.1000, take_profit=1.1040, direction=Direction.LONG) == 1.1020


def test_compute_tp1_is_halfway_by_default_short():
    assert compute_tp1(entry_price=1.1000, take_profit=1.0960, direction=Direction.SHORT) == 1.0980


def test_compute_tp1_respects_custom_fraction():
    assert compute_tp1(1.1000, 1.1040, Direction.LONG, tp1_fraction=0.25) == 1.1010


def test_long_takes_partial_profit_once_tp1_reached():
    assert should_take_partial_profit(Direction.LONG, entry_price=1.1000, take_profit=1.1040, current_price=1.1020)


def test_long_does_not_take_partial_profit_before_tp1():
    assert not should_take_partial_profit(Direction.LONG, entry_price=1.1000, take_profit=1.1040, current_price=1.1015)


def test_short_takes_partial_profit_once_tp1_reached():
    assert should_take_partial_profit(Direction.SHORT, entry_price=1.1000, take_profit=1.0960, current_price=1.0980)


def test_short_does_not_take_partial_profit_before_tp1():
    assert not should_take_partial_profit(Direction.SHORT, entry_price=1.1000, take_profit=1.0960, current_price=1.0990)


def test_returns_false_when_no_take_profit_set():
    assert not should_take_partial_profit(Direction.LONG, entry_price=1.1000, take_profit=0.0, current_price=1.1050)


def test_returns_false_for_neutral_direction():
    assert not should_take_partial_profit(Direction.NEUTRAL, entry_price=1.1000, take_profit=1.1040, current_price=1.1050)


def test_long_needs_breakeven_when_sl_still_below_entry():
    assert should_move_to_breakeven(Direction.LONG, entry_price=1.1000, current_sl=1.0980)


def test_long_leaves_sl_alone_when_already_at_or_past_breakeven():
    assert not should_move_to_breakeven(Direction.LONG, entry_price=1.1000, current_sl=1.1000)
    # El trailing ya lo mejoro mas alla de breakeven -> no aflojar.
    assert not should_move_to_breakeven(Direction.LONG, entry_price=1.1000, current_sl=1.1010)


def test_short_needs_breakeven_when_sl_still_above_entry():
    assert should_move_to_breakeven(Direction.SHORT, entry_price=1.1000, current_sl=1.1020)


def test_short_leaves_sl_alone_when_already_at_or_past_breakeven():
    assert not should_move_to_breakeven(Direction.SHORT, entry_price=1.1000, current_sl=1.1000)
    assert not should_move_to_breakeven(Direction.SHORT, entry_price=1.1000, current_sl=1.0990)
