import pandas as pd

from trader.config import SessionsConfig, SessionWindowConfig
from trader.sessions import classify_session

_CONFIG = SessionsConfig(
    asia=SessionWindowConfig(timezone="Asia/Tokyo", start_hour=9, end_hour=18),
    london=SessionWindowConfig(timezone="Europe/London", start_hour=8, end_hour=17),
    new_york=SessionWindowConfig(timezone="America/New_York", start_hour=8, end_hour=17),
    killzone_london=SessionWindowConfig(timezone="Europe/London", start_hour=7, end_hour=10),
    killzone_new_york=SessionWindowConfig(timezone="America/New_York", start_hour=8, end_hour=11),
)


def test_london_session_shifts_with_dst():
    winter = pd.Timestamp("2026-01-15 07:30", tz="UTC")  # GMT, London opens 08:00 UTC
    summer = pd.Timestamp("2026-07-15 07:30", tz="UTC")  # BST, London opens 07:00 UTC

    assert classify_session(winter, _CONFIG).london is False
    assert classify_session(summer, _CONFIG).london is True


def test_new_york_session_shifts_with_dst():
    winter = pd.Timestamp("2026-01-15 12:30", tz="UTC")  # EST, NY opens 13:00 UTC
    summer = pd.Timestamp("2026-07-15 12:30", tz="UTC")  # EDT, NY opens 12:00 UTC

    assert classify_session(winter, _CONFIG).new_york is False
    assert classify_session(summer, _CONFIG).new_york is True


def test_asia_session_is_dst_invariant():
    winter = pd.Timestamp("2026-01-15 05:00", tz="UTC")
    summer = pd.Timestamp("2026-07-15 05:00", tz="UTC")

    assert classify_session(winter, _CONFIG).asia is True
    assert classify_session(summer, _CONFIG).asia is True


def test_overlap_and_active_labels():
    overlap_instant = pd.Timestamp("2026-07-15 13:00", tz="UTC")  # summer: London 07-16, NY 12-21 UTC
    state = classify_session(overlap_instant, _CONFIG)

    assert state.london is True
    assert state.new_york is True
    assert state.overlap_london_ny is True
    assert state.active_labels == ["Londres", "NY"]


def test_outside_all_sessions():
    quiet = pd.Timestamp("2026-01-15 22:30", tz="UTC")
    state = classify_session(quiet, _CONFIG)
    assert state.any_active is False
