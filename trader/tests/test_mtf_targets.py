from __future__ import annotations

from trader.mtf_strategies.targets import compute_levels_with_priority_tp2, select_tp2


def test_select_tp2_picks_the_first_valid_candidate_in_priority_order_long():
    # entry=100, LONG needs a level ABOVE entry -- first candidate (105) qualifies
    priority = [105.0, 110.0, 120.0]
    assert select_tp2("long", priority, entry=100.0) == 105.0


def test_select_tp2_falls_back_to_next_level_when_first_is_unavailable():
    # first priority (PDH-equivalent) is None -- not "not attempted", skipped
    # to the next in order, not silently the highest priority regardless
    priority = [None, 110.0, 120.0]
    assert select_tp2("long", priority, entry=100.0) == 110.0


def test_select_tp2_skips_a_candidate_on_the_wrong_side_of_entry():
    # first candidate is BELOW entry for a LONG (not a valid target) -- must
    # be skipped even though it's not None, same as an unavailable level
    priority = [95.0, 108.0]
    assert select_tp2("long", priority, entry=100.0) == 108.0


def test_select_tp2_short_direction_mirrors_long():
    priority = [None, 90.0, 80.0]
    assert select_tp2("short", priority, entry=100.0) == 90.0


def test_select_tp2_none_when_no_candidate_qualifies():
    assert select_tp2("long", [None, None], entry=100.0) is None
    assert select_tp2("long", [95.0, 90.0], entry=100.0) is None  # both below entry, wrong side


def test_compute_levels_with_priority_tp2_respects_min_rr_gate():
    # entry=100, sl=99 (risk=1), tp2=101 (reward=1) -> rr=1, below min_rr=2 -> None
    result = compute_levels_with_priority_tp2("long", entry=100.0, structural_stop=99.0, priority_levels=[101.0], min_rr=2.0)
    assert result is None


def test_compute_levels_with_priority_tp2_returns_levels_when_rr_sufficient():
    # entry=100, sl=99 (risk=1), tp2=103 (reward=3) -> rr=3 >= min_rr=2
    result = compute_levels_with_priority_tp2("long", entry=100.0, structural_stop=99.0, priority_levels=[103.0], min_rr=2.0)
    assert result is not None
    assert result.tp == 103.0
    assert result.risk_reward == 3.0


def test_compute_levels_with_priority_tp2_none_when_no_tp2_available():
    result = compute_levels_with_priority_tp2("long", entry=100.0, structural_stop=99.0, priority_levels=[None], min_rr=2.0)
    assert result is None
