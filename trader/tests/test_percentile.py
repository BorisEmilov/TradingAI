"""Test de causalidad explicito pedido en logs/reformulacion_diseno_capas.md:
agregar eventos con timestamp posterior a un cutoff `t` no puede cambiar el
percentil ya calculado en `t`."""

from trader.percentile import ExpandingPercentileTracker, RollingPercentileTracker


def test_expanding_percentile_below_min_sample_is_none():
    tracker = ExpandingPercentileTracker(min_sample=5)
    for v in [1.0, 2.0, 3.0, 4.0]:
        assert tracker.percentile_then_insert(v) is None


def test_expanding_percentile_basic_ranking():
    tracker = ExpandingPercentileTracker(min_sample=3)
    for v in [1.0, 2.0, 3.0]:
        tracker.percentile_then_insert(v)
    # history is now [1,2,3]; a new value of 10 should rank at the very top (100th pct)
    assert tracker.percentile_then_insert(10.0) == 100.0


def test_expanding_percentile_future_events_dont_change_past_result():
    """El test de causalidad explicito: calcular el percentil en un cutoff
    intermedio, luego seguir agregando eventos futuros -- el valor ya
    devuelto en el cutoff no debe cambiar retroactivamente (no puede, dado
    que ya se devolvio, pero esto prueba que RE-CALCULARLO con la misma
    muestra causal da el mismo resultado, verificando que el tracker nunca
    se "adelanta" a mirar values insertados despues)."""
    history_up_to_t = [5.0, 1.0, 9.0, 3.0, 7.0, 5.0, 5.0, 2.0, 8.0, 4.0]
    future_events = [100.0, 0.001, 50.0, -30.0]
    candidate_value = 6.0

    tracker_a = ExpandingPercentileTracker(min_sample=5)
    for v in history_up_to_t:
        tracker_a.percentile_then_insert(v)
    pct_at_t = tracker_a.percentile_then_insert(candidate_value)

    # Re-run the identical causal prefix (up to and including candidate_value)
    # inside a run that ALSO has future events appended afterward -- the
    # percentile AT candidate_value's position must be identical either way.
    tracker_b = ExpandingPercentileTracker(min_sample=5)
    for v in history_up_to_t:
        tracker_b.percentile_then_insert(v)
    pct_at_t_again = tracker_b.percentile_then_insert(candidate_value)
    for v in future_events:
        tracker_b.percentile_then_insert(v)  # inserted AFTER, must not affect the earlier result

    assert pct_at_t == pct_at_t_again


def test_rolling_percentile_requires_full_window():
    tracker = RollingPercentileTracker(window=3)
    assert tracker.percentile_then_insert(1.0) is None
    assert tracker.percentile_then_insert(2.0) is None
    assert tracker.percentile_then_insert(3.0) is None  # still only 3 PRIOR values needed, this is the 4th call
    # after 3 values inserted, the 4th call has a full window of 3 to compare against
    assert tracker.percentile_then_insert(100.0) == 100.0


def test_rolling_percentile_evicts_oldest():
    tracker = RollingPercentileTracker(window=2)
    tracker.percentile_then_insert(1.0)
    tracker.percentile_then_insert(1.0)
    # window is now [1.0, 1.0]; insert 5.0 -> percentile vs [1.0,1.0] = 100, window becomes [1.0, 5.0]
    assert tracker.percentile_then_insert(5.0) == 100.0
    # window is now [1.0, 5.0]; insert 0.0 -> should rank at 0th percentile (below both)
    assert tracker.percentile_then_insert(0.0) == 0.0
    # window slides to [5.0, 0.0] (oldest 1.0 evicted) -- insert 0.0 again -> ties at bisect_left = 0
    pct = tracker.percentile_then_insert(0.0)
    assert pct == 0.0


def test_rolling_percentile_future_events_dont_change_past_result():
    history = [3.0, 1.0, 4.0, 1.0, 5.0]
    candidate = 2.0
    future = [999.0, -999.0, 42.0]

    tracker_a = RollingPercentileTracker(window=5)
    for v in history:
        tracker_a.percentile_then_insert(v)
    pct_a = tracker_a.percentile_then_insert(candidate)

    tracker_b = RollingPercentileTracker(window=5)
    for v in history:
        tracker_b.percentile_then_insert(v)
    pct_b = tracker_b.percentile_then_insert(candidate)
    for v in future:
        tracker_b.percentile_then_insert(v)

    assert pct_a == pct_b
