"""Structured signal log (JSON Lines) -- one line per analysis outcome,
including NO-signal reasons, so the confluence minimum and invalidation
criteria can later be tuned against real data (per the strategy spec)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trader.pipeline.engine import NoSignal
from trader.signal import TradingSignal


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _append(record: dict, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")


def log_signal(signal: TradingSignal, path: str) -> None:
    _append(
        {
            "generated_at": signal.generated_at.isoformat(),
            "symbol": signal.symbol,
            "signal": True,
            "direction": signal.direction,
            "bias_1d": signal.bias_1d,
            "session": signal.session.active_labels,
            "poi_kind": signal.poi.kind,
            "poi_high": signal.poi.price_high,
            "poi_low": signal.poi.price_low,
            "confirmation_kind": signal.confirmation.kind,
            "confluence_families": sorted(signal.confluences.families),
            "confluence_timeframes": sorted(signal.confluences.timeframes),
            "entry": signal.levels.entry,
            "sl": signal.levels.sl,
            "tp": signal.levels.tp,
            "risk_reward": signal.levels.risk_reward,
        },
        path,
    )


def log_no_signal(no_signal: NoSignal, symbol: str, as_of: pd.Timestamp, path: str) -> None:
    _append(
        {
            "generated_at": as_of.isoformat(),
            "symbol": symbol,
            "signal": False,
            "stage": no_signal.stage,
            "reason": no_signal.reason,
        },
        path,
    )
