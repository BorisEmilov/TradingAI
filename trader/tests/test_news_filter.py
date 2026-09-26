import gc

import pandas as pd

from trader.config import NewsEventConfig, NewsFilterConfig
from trader.news_filter import _parsed_manual_events, is_high_impact_news_window


def _config(**overrides) -> NewsFilterConfig:
    base = dict(enabled=True, buffer_minutes_before=30, buffer_minutes_after=60, nfp_auto=True, manual_events=[])
    base.update(overrides)
    return NewsFilterConfig(**base)


def test_nfp_first_friday_of_month_is_blocked():
    # 2025-08-01 is a Friday (and the first Friday of August 2025) -- the
    # exact date the M15_confirmation audit's 2 extreme-drift cases landed on.
    nfp_utc = pd.Timestamp("2025-08-01T12:30:00", tz="UTC")
    config = _config()

    blocked, label = is_high_impact_news_window(nfp_utc, config)

    assert blocked
    assert "NFP" in label


def test_nfp_buffer_edges_respected():
    nfp_utc = pd.Timestamp("2025-08-01T12:30:00", tz="UTC")
    config = _config(buffer_minutes_before=30, buffer_minutes_after=60)

    just_inside_before = nfp_utc - pd.Timedelta(minutes=30)
    just_inside_after = nfp_utc + pd.Timedelta(minutes=60)
    just_outside_before = nfp_utc - pd.Timedelta(minutes=31)
    just_outside_after = nfp_utc + pd.Timedelta(minutes=61)

    assert is_high_impact_news_window(just_inside_before, config)[0]
    assert is_high_impact_news_window(just_inside_after, config)[0]
    assert not is_high_impact_news_window(just_outside_before, config)[0]
    assert not is_high_impact_news_window(just_outside_after, config)[0]


def test_nfp_disabled_by_nfp_auto_flag():
    nfp_utc = pd.Timestamp("2025-08-01T12:30:00", tz="UTC")
    config = _config(nfp_auto=False)

    blocked, label = is_high_impact_news_window(nfp_utc, config)

    assert not blocked
    assert label is None


def test_manual_event_blocks_within_buffer():
    event_ts = "2025-07-30T18:00:00+00:00"  # FOMC, from config.yaml
    config = _config(nfp_auto=False, manual_events=[NewsEventConfig(timestamp_utc=event_ts, label="FOMC")])

    inside = pd.Timestamp(event_ts) + pd.Timedelta(minutes=45)
    outside = pd.Timestamp(event_ts) + pd.Timedelta(minutes=90)

    blocked_inside, label = is_high_impact_news_window(inside, config)
    blocked_outside, _ = is_high_impact_news_window(outside, config)

    assert blocked_inside
    assert "FOMC" in label
    assert not blocked_outside


def test_a_normal_tuesday_afternoon_is_not_blocked():
    ordinary = pd.Timestamp("2025-08-05T09:00:00", tz="UTC")  # Tuesday, no NFP/FOMC nearby
    config = _config()

    blocked, label = is_high_impact_news_window(ordinary, config)

    assert not blocked
    assert label is None


def test_disabled_config_never_blocks_even_on_an_nfp_timestamp():
    nfp_utc = pd.Timestamp("2025-08-01T12:30:00", tz="UTC")
    config = _config(enabled=False)

    blocked, label = is_high_impact_news_window(nfp_utc, config)

    assert not blocked
    assert label is None


def test_nfp_is_correct_across_a_dst_transition():
    # NFP is always 8:30 America/New_York -- EST (UTC-5) in Nov/Dec/Jan/Feb,
    # EDT (UTC-4) most of the rest of the year. A fixed UTC-offset
    # implementation would be wrong for half the year; this checks both.
    winter_nfp = pd.Timestamp("2025-01-03T13:30:00", tz="UTC")  # first Friday of Jan 2025, EST -> 13:30 UTC
    summer_nfp = pd.Timestamp("2025-08-01T12:30:00", tz="UTC")  # first Friday of Aug 2025, EDT -> 12:30 UTC
    config = _config()

    assert is_high_impact_news_window(winter_nfp, config)[0]
    assert is_high_impact_news_window(summer_nfp, config)[0]


def test_manual_events_cache_keyed_by_content_not_object_identity():
    """BUG FIX 2026-09-23: `_parsed_manual_events` cacheaba por `id(config)` --
    id() se puede reciclar cuando el objeto anterior ya fue recolectado
    (frecuente acá: un `NewsFilterConfig` nuevo por test). Un config con
    `manual_events` distinto podía heredar la caché vieja del id recién
    liberado. Fuerza la recolección entre dos configs de contenido distinto
    -- si la caché siguiera indexada por identidad, esto podría devolver el
    resultado del primero para el segundo."""
    first = NewsFilterConfig(manual_events=[NewsEventConfig(timestamp_utc="2025-07-30T18:00:00+00:00", label="FOMC")])
    first_result = _parsed_manual_events(first)
    del first
    gc.collect()

    second = NewsFilterConfig(manual_events=[])
    second_result = _parsed_manual_events(second)

    assert first_result != second_result
    assert second_result == ()
