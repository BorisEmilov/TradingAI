from trader.risk.management import PositionAction, PositionPlan, evaluate_position


def _plan():
    return PositionPlan(direction="long", entry=100.0, sl=99.0, tp=103.0)


def test_hold_before_partial_threshold():
    plan = _plan()
    action = evaluate_position(plan, current_price=101.0, partial_at_progress_pct=0.5, htf_invalidation=False)
    assert action == PositionAction.HOLD


def test_take_partial_at_threshold():
    plan = _plan()  # halfway to TP = 101.5
    action = evaluate_position(plan, current_price=101.5, partial_at_progress_pct=0.5, htf_invalidation=False)
    assert action == PositionAction.TAKE_PARTIAL_AND_MOVE_SL_BE


def test_close_full_on_invalidation_before_partial():
    plan = _plan()
    action = evaluate_position(plan, current_price=100.2, partial_at_progress_pct=0.5, htf_invalidation=True)
    assert action == PositionAction.CLOSE_FULL


def test_close_breakeven_on_invalidation_after_partial():
    plan = _plan()
    plan.partial_taken = True
    action = evaluate_position(plan, current_price=102.0, partial_at_progress_pct=0.5, htf_invalidation=True)
    assert action == PositionAction.CLOSE_BREAKEVEN


def test_short_direction_progress():
    plan = PositionPlan(direction="short", entry=100.0, sl=101.0, tp=94.0)
    action = evaluate_position(plan, current_price=97.0, partial_at_progress_pct=0.5, htf_invalidation=False)
    assert action == PositionAction.TAKE_PARTIAL_AND_MOVE_SL_BE
