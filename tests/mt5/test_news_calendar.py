from datetime import datetime, timezone

from tradingai.mt5.news_calendar import is_news_blackout


def test_nfp_blocks_around_known_date_edt():
    # NFP de septiembre 2026 (verano/EDT): primer viernes = 2026-09-04, 8:30 ET = 12:30 UTC.
    config = {"minutes_before": 15, "minutes_after": 15}
    just_before = datetime(2026, 9, 4, 12, 20, tzinfo=timezone.utc)
    just_after = datetime(2026, 9, 4, 12, 40, tzinfo=timezone.utc)
    assert is_news_blackout(just_before, config)
    assert is_news_blackout(just_after, config)


def test_nfp_blocks_around_known_date_est():
    # NFP de enero 2026 (invierno/EST): primer viernes = 2026-01-02, 8:30 ET = 13:30 UTC.
    config = {"minutes_before": 15, "minutes_after": 15}
    at_event = datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc)
    assert is_news_blackout(at_event, config)


def test_outside_nfp_window_not_blocked():
    config = {"minutes_before": 15, "minutes_after": 15}
    far_away = datetime(2026, 9, 4, 15, 0, tzinfo=timezone.utc)
    assert not is_news_blackout(far_away, config)


def test_manual_event_blocks_its_window():
    config = {
        "minutes_before": 10,
        "minutes_after": 10,
        "nfp": {"enabled": False},
        "manual_events": [{"date": "2026-09-17", "time_utc": "18:00", "label": "FOMC"}],
    }
    inside = datetime(2026, 9, 17, 18, 5, tzinfo=timezone.utc)
    outside = datetime(2026, 9, 17, 19, 0, tzinfo=timezone.utc)
    assert is_news_blackout(inside, config)
    assert not is_news_blackout(outside, config)


def test_disabled_calendar_never_blocks():
    config = {"enabled": False, "manual_events": [{"date": "2026-09-17", "time_utc": "18:00", "label": "FOMC"}]}
    at_event = datetime(2026, 9, 17, 18, 0, tzinfo=timezone.utc)
    assert not is_news_blackout(at_event, config)


def test_nfp_disabled_individually():
    config = {"nfp": {"enabled": False}}
    at_nfp = datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc)
    assert not is_news_blackout(at_nfp, config)


def test_none_config_never_blocks():
    assert not is_news_blackout(datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc), None)
