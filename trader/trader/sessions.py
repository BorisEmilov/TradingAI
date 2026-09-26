"""Market session classification (Asia/London/New York) with DST handling.

Session hours are defined in *local exchange time* (config.yaml) and
converted to UTC per calendar date via IANA tzdata, so London/New York
daylight-saving transitions shift the UTC window automatically instead of
using a fixed UTC offset that would be wrong half the year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from trader.config import SessionsConfig, SessionWindowConfig


def _window_utc(window: SessionWindowConfig, on_date: datetime) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz = ZoneInfo(window.timezone)
    local_date = on_date.astimezone(tz).date()
    start_local = datetime(
        local_date.year, local_date.month, local_date.day, window.start_hour, tzinfo=tz
    )
    end_local = datetime(
        local_date.year, local_date.month, local_date.day, window.end_hour, tzinfo=tz
    )
    if window.end_hour <= window.start_hour:
        end_local = end_local + pd.Timedelta(days=1)
    return pd.Timestamp(start_local).tz_convert("UTC"), pd.Timestamp(end_local).tz_convert("UTC")


@dataclass(frozen=True)
class SessionState:
    timestamp: pd.Timestamp
    asia: bool
    london: bool
    new_york: bool
    killzone_london: bool
    killzone_new_york: bool

    @property
    def overlap_london_ny(self) -> bool:
        return self.london and self.new_york

    @property
    def any_active(self) -> bool:
        return self.asia or self.london or self.new_york

    @property
    def active_labels(self) -> list[str]:
        labels = []
        if self.asia:
            labels.append("Asia")
        if self.london:
            labels.append("Londres")
        if self.new_york:
            labels.append("NY")
        return labels


def _in_window(ts: pd.Timestamp, window: SessionWindowConfig) -> bool:
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    # A window can start "yesterday" locally and still cover `ts` (e.g. checking
    # 00:30 UTC against a session whose local window crosses midnight UTC), so
    # both today's and yesterday's instance of the window are checked.
    for day_offset in (0, -1):
        probe = ts + pd.Timedelta(days=day_offset)
        start, end = _window_utc(window, probe.to_pydatetime())
        if start <= ts < end:
            return True
    return False


def classify_session(ts: pd.Timestamp, config: SessionsConfig) -> SessionState:
    return SessionState(
        timestamp=ts,
        asia=_in_window(ts, config.asia),
        london=_in_window(ts, config.london),
        new_york=_in_window(ts, config.new_york),
        killzone_london=_in_window(ts, config.killzone_london),
        killzone_new_york=_in_window(ts, config.killzone_new_york),
    )
