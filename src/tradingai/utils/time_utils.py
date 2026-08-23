"""Utilidades de tiempo/timeframes compartidas por el modulo AI y el modulo MT5."""

from __future__ import annotations

from datetime import timedelta

# Minutos por timeframe, usados para resamplear velas y alinear features.
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Timeframe desconocido: {timeframe}")
    return timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
