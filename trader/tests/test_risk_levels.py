from trader.risk.levels import compute_trade_levels


def test_long_meets_minimum_rr():
    levels = compute_trade_levels("long", entry=100.0, structural_stop=99.0, structural_target=103.0, min_rr=2.0)
    assert levels is not None
    assert levels.risk_reward == 3.0


def test_long_below_minimum_rr_rejected():
    levels = compute_trade_levels("long", entry=100.0, structural_stop=99.0, structural_target=101.5, min_rr=2.0)
    assert levels is None


def test_short_meets_minimum_rr():
    levels = compute_trade_levels("short", entry=100.0, structural_stop=101.0, structural_target=96.0, min_rr=2.0)
    assert levels is not None
    assert levels.risk_reward == 4.0


def test_zero_or_negative_risk_rejected():
    assert compute_trade_levels("long", entry=100.0, structural_stop=100.0, structural_target=110.0, min_rr=1.0) is None
    assert compute_trade_levels("long", entry=100.0, structural_stop=101.0, structural_target=110.0, min_rr=1.0) is None


def test_zero_or_negative_reward_rejected():
    assert compute_trade_levels("long", entry=100.0, structural_stop=99.0, structural_target=100.5, min_rr=0.01) is not None
    assert compute_trade_levels("long", entry=100.0, structural_stop=99.0, structural_target=100.0, min_rr=0.01) is None
