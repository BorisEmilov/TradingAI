"""prompt-piloto-forward-testing.md: isolated forward-testing pilot for
rsi14_XPTUSD (the only one of 4 thin-VALIDATION-sample candidates that passed
the concentration check -- see logs/pilot_rsi_xpt_prereg_2026-09-20.md, written
BEFORE this ran, rules frozen there).

Self-contained on purpose -- does NOT touch trader/pipeline/, trader/risk/, or
any other part of the ICT system. A single long-running process: ONE gateway
login at startup (kept alive via client.refresh(), never re-login in a loop),
polls hourly, persists its state to a local JSON file so it can resume
correctly if restarted. Demo-only by construction -- the gateway itself
(require_trading_allowed()) refuses a real account unless PYGW_ALLOW_REAL=1.

Exit is pure TIME (10 trading days), matching exactly what was backtested --
the only deliberate deviation is a wide 5xATR safety stop, documented in the
pre-registration, never part of the tested signal logic.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.config import load_config
from trader.detectors.indicators import atr
from trader.gateway_client import PythonGetawayClient
from trader.quant.factors import _rsi

SYMBOL = "XPTUSD"
RSI_PERIOD = 14
OVERSOLD, OVERBOUGHT = 30.0, 70.0
HOLD_TRADING_DAYS = 10
VOLUME = 0.1
SAFETY_STOP_ATR_MULT = 5.0
MAX_TRADES = 10
MAX_MONTHS = 12
POLL_SECONDS = 3600  # 1h -- frequent enough to catch a new D1 close promptly, not wasteful

STATE_PATH = Path(__file__).resolve().parent.parent / "logs" / "pilot_rsi_xpt_state.json"
TRADES_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "pilot_rsi_xpt_trades.jsonl"
EVENTS_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "pilot_rsi_xpt_events.jsonl"


@dataclass
class OpenPosition:
    ticket: int
    direction: str  # "long" | "short"
    entry_time: str
    entry_price: float
    entry_bar_timestamp: str  # timestamp of the D1 bar that triggered entry -- counting new
    # closed bars > this timestamp on each tick is what tracks the 10-trading-day hold, NOT a
    # raw len(candles) count: candles() always returns the last N rows of a rolling window, so
    # len(candles) stays ~constant tick to tick and can't be used as an elapsed-bars counter.


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(kind: str, **fields) -> None:
    row = {"ts": _now_iso(), "kind": kind, **fields}
    with open(EVENTS_LOG_PATH, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"[{row['ts']}] {kind}: {fields}")


def _log_trade(row: dict) -> None:
    with open(TRADES_LOG_PATH, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"started_at": _now_iso(), "closed_trades": 0, "open_position": None}


def _save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _months_elapsed(started_at: str) -> float:
    start = datetime.fromisoformat(started_at)
    return (datetime.now(timezone.utc) - start).days / 30.44


def _check_stop_criteria(state: dict) -> str | None:
    if state["closed_trades"] >= MAX_TRADES:
        return f"MAX_TRADES alcanzado ({state['closed_trades']}/{MAX_TRADES})"
    months = _months_elapsed(state["started_at"])
    if months >= MAX_MONTHS:
        return f"MAX_MONTHS alcanzado ({months:.1f}/{MAX_MONTHS})"
    return None


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()
    _log_event("pilot_started", symbol=SYMBOL, rsi_period=RSI_PERIOD, volume=VOLUME, max_trades=MAX_TRADES, max_months=MAX_MONTHS)

    state = _load_state()
    last_refresh = time.time()

    try:
        while True:
            stop_reason = _check_stop_criteria(state)
            if stop_reason:
                if state["open_position"]:
                    _log_event("stop_criteria_reached_closing_open_position", reason=stop_reason)
                    _close_position(client, state)
                _log_event("pilot_finished", reason=stop_reason, closed_trades=state["closed_trades"])
                break

            if time.time() - last_refresh > 1200:  # refresh well before any plausible session TTL, still not a re-login
                client.refresh()
                last_refresh = time.time()

            try:
                _tick(client, state)
            except Exception as exc:  # noqa: BLE001 -- one bad tick must never kill a months-long pilot
                _log_event("tick_error", error=str(exc), trace=traceback.format_exc()[-2000:])

            _save_state(state)
            time.sleep(POLL_SECONDS)
    finally:
        client.logout()
        _log_event("session_closed", closed_trades=state["closed_trades"])


def _tick(client: PythonGetawayClient, state: dict) -> None:
    candles = client.candles(SYMBOL, "D1", 200, timeout=60)
    if len(candles) < RSI_PERIOD + 5:
        _log_event("insufficient_data", n=len(candles))
        return

    latest = candles.iloc[-1]
    rsi_series = _rsi(candles["close"], RSI_PERIOD)
    latest_rsi = float(rsi_series[-1])

    if state["open_position"] is not None:
        pos = state["open_position"]
        entry_ts = pd.Timestamp(pos["entry_bar_timestamp"])
        bars_held = int((candles["timestamp"] > entry_ts).sum())
        _log_event("check_open_position", ticket=pos["ticket"], bars_held=bars_held, target=HOLD_TRADING_DAYS, latest_rsi=latest_rsi)
        if bars_held >= HOLD_TRADING_DAYS:
            _close_position(client, state)
        return

    _log_event("check_no_position", latest_close=float(latest["close"]), latest_rsi=latest_rsi)
    if latest_rsi < OVERSOLD:
        _open_position(client, state, candles, "long", latest_rsi)
    elif latest_rsi > OVERBOUGHT:
        _open_position(client, state, candles, "short", latest_rsi)


def _open_position(client: PythonGetawayClient, state: dict, candles: pd.DataFrame, direction: str, rsi_value: float) -> None:
    info = client.symbol_info(SYMBOL, timeout=30)
    last_price = client.last_price(SYMBOL)
    atr_series = atr(candles, period=14)
    latest_atr = float(atr_series.iloc[-1])
    stop_distance = SAFETY_STOP_ATR_MULT * latest_atr
    sl = last_price - stop_distance if direction == "long" else last_price + stop_distance

    result = client.open_position(
        symbol=SYMBOL, direction=direction, volume=VOLUME, sl=round(sl, info["digits"]), tp=0.0,
        comment=f"pilot_rsi14_v1_{rsi_value:.0f}",
    )
    _log_event("open_attempt", direction=direction, rsi=rsi_value, price=last_price, sl=sl, result=result)

    if not result.get("success"):
        _log_event("open_failed", retcode=result.get("retcode_name"))
        return

    pos = OpenPosition(
        ticket=result["position"], direction=direction, entry_time=_now_iso(),
        entry_price=result.get("price", last_price), entry_bar_timestamp=str(candles["timestamp"].iloc[-1]),
    )
    state["open_position"] = asdict(pos)
    _log_event("position_opened", **state["open_position"])


def _close_position(client: PythonGetawayClient, state: dict) -> None:
    pos = state["open_position"]
    result = client.close_position(pos["ticket"])
    _log_event("close_attempt", ticket=pos["ticket"], result=result)

    exit_price = result.get("price")
    trade_row = {
        "ticket": pos["ticket"], "direction": pos["direction"],
        "entry_time": pos["entry_time"], "entry_price": pos["entry_price"],
        "exit_time": _now_iso(), "exit_price": exit_price,
        "success": result.get("success"), "raw_result": result,
    }
    _log_trade(trade_row)
    state["closed_trades"] += 1
    state["open_position"] = None
    _log_event("position_closed", trade=trade_row, total_closed=state["closed_trades"])


if __name__ == "__main__":
    main()
