"""High-impact news window filter.

The original strategy spec (prompt-estrategia-trading-agente.md, section on
entry conditions) says: "No operar ... en ventanas de noticias de alto
impacto (NFP, decisiones de tipos, CPI, etc.)". This was never implemented
anywhere in the pipeline until now -- found missing while auditing why
loosening M15_confirmation's freshness window exposed 2 extreme-drift
example cases (~135-150 pips) that both landed on 2025-08-01, the first
Friday of August -- an NFP release date. See prompt-implementar-m15-n2.md.

Covers, for now:
  - NFP (US Non-Farm Payrolls): released the first Friday of every month at
    8:30 AM America/New_York (handled via IANA tzdata, not a fixed UTC
    offset, so it's correct across the EST/EDT transition). Computed
    programmatically -- no manual data entry, so it covers the entire
    history/future with no maintenance and no risk of a wrong hardcoded date.
  - FOMC rate decisions: a curated list in config.yaml
    (`news_filter.manual_events`), 2:00 PM America/New_York on the second day
    of each meeting, sourced from the Federal Reserve's own published
    calendar (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm,
    verified 2026-09-17) and converted to UTC by hand once, not hardcoded per
    date guesswork. There is no machine-readable Fed feed, so this list needs
    a manual update whenever new meeting dates are announced.

NOT covered yet (a documented gap, not a silent one):
  - CPI releases: the BLS schedule is public (bls.gov/schedule/news_release/
    cpi.htm) but assembling a fully-verified multi-year list was out of scope
    for this pass -- the 2 extreme-drift audit cases were both NFP, not CPI,
    so there's no concrete evidence yet that CPI matters as much for this
    system. Worth adding if/when CPI-correlated drift shows up in a real
    backtest.
  - Non-US central bank decisions (ECB/BOE/BOJ), even though they're directly
    relevant to EURUSD/GBPUSD/USDJPY specifically. Same reasoning: no
    concrete evidence from the audit implicated them, so a partially-sourced
    multi-bank calendar wasn't worth the transcription-error risk right now.

Both gaps are natural follow-ups, not something to quietly build in later
without saying so.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

from trader.config import NewsFilterConfig

_NFP_HOUR_ET = 8
_NFP_MINUTE_ET = 30
_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


@lru_cache(maxsize=256)
def _first_friday_nfp_utc(year: int, month: int) -> pd.Timestamp:
    first_of_month = datetime(year, month, 1)
    days_ahead = (4 - first_of_month.weekday()) % 7  # Monday=0 ... Friday=4
    first_friday = first_of_month + timedelta(days=days_ahead)
    local = datetime(
        first_friday.year, first_friday.month, first_friday.day, _NFP_HOUR_ET, _NFP_MINUTE_ET, tzinfo=_ET
    )
    return pd.Timestamp(local.astimezone(_UTC))


def _nearby_nfp_events(as_of: pd.Timestamp) -> list[pd.Timestamp]:
    # This month plus both neighbors -- cheap, and correct even for a buffer
    # window straddling a month boundary (never happens in practice since the
    # first Friday is never in the first few days, but it's free to handle).
    events = []
    for delta_months in (-1, 0, 1):
        month = as_of.month + delta_months
        year = as_of.year
        while month < 1:
            month += 12
            year -= 1
        while month > 12:
            month -= 12
            year += 1
        events.append(_first_friday_nfp_utc(year, month))
    return events


# `config.manual_events` is a small, frozen list re-used across an entire
# backtest run (tens of thousands of `is_high_impact_news_window` calls in
# the hot loop) -- re-parsing every ISO string on every call was a real,
# measurable cost at that call volume.
#
# BUG FIX 2026-09-23 (found auditando algo no relacionado, ver
# project_news_filter_id_cache_bug_2026-09-23 en memoria): esto estaba
# cacheado por `id(config)`. `NewsFilterConfig` es un frozen dataclass, pero
# `manual_events` es una `list` (no hashable) -- eso ya impedía usar
# `@lru_cache` directo sobre `config`, y la alternativa elegida (cachear por
# id de objeto) es insegura: `id()` solo es único entre objetos VIVOS al
# mismo tiempo, CPython reutiliza la dirección de memoria de un objeto ya
# recolectado. Un `NewsFilterConfig` de vida corta (frecuente en tests, que
# construyen uno nuevo por caso) puede terminar con el mismo `id()` que uno
# posterior con `manual_events` DISTINTO -- la caché devuelve entonces la
# lista vieja (a veces vacía) en vez de volver a parsear, y un evento real
# nunca bloquea. Reproducido de forma intermitente en
# `tests/test_news_filter.py::test_manual_event_blocks_within_buffer` al
# correr la suite completa (pasaba en aislamiento, fallaba ~1 de cada 3-4
# corridas junto al resto). Clave por CONTENIDO (timestamp+label de cada
# evento), no por identidad -- correcto sin importar el ciclo de vida del
# objeto, y de paso comparte caché entre configs distintas con el mismo
# `manual_events`.
_manual_events_cache: dict[tuple[tuple[str, str], ...], tuple[tuple[pd.Timestamp, str], ...]] = {}


def _parsed_manual_events(config: NewsFilterConfig) -> tuple[tuple[pd.Timestamp, str], ...]:
    key = tuple((event.timestamp_utc, event.label) for event in config.manual_events)
    cached = _manual_events_cache.get(key)
    if cached is not None:
        return cached
    parsed = []
    for event in config.manual_events:
        ts = pd.Timestamp(event.timestamp_utc)
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        parsed.append((ts, event.label))
    result = tuple(parsed)
    _manual_events_cache[key] = result
    return result


def is_high_impact_news_window(as_of: pd.Timestamp, config: NewsFilterConfig) -> tuple[bool, str | None]:
    """Returns (blocked, label). `as_of` must be UTC-aware (or naive-UTC)."""
    if not config.enabled:
        return False, None

    as_of = as_of.tz_convert("UTC") if as_of.tzinfo else as_of.tz_localize("UTC")
    before = pd.Timedelta(minutes=config.buffer_minutes_before)
    after = pd.Timedelta(minutes=config.buffer_minutes_after)

    if config.nfp_auto:
        for event_ts in _nearby_nfp_events(as_of):
            if event_ts - before <= as_of <= event_ts + after:
                return True, f"NFP {event_ts.isoformat()}"

    for event_ts, label in _parsed_manual_events(config):
        if event_ts - before <= as_of <= event_ts + after:
            return True, f"{label} {event_ts.isoformat()}"

    return False, None
