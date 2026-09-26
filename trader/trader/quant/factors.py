"""prompt-fase-cuantitativa.md: the 3 factor families that actually generate
tests (momentum, mean-reversion, session/weekday seasonality). Volatility
regime (factor C) is deliberately NOT here as a standalone battery -- it's
applied only to EXPLORATION survivors, in scripts/run_quant_battery.py.

Every function returns arrays aligned to `candles` bar positions within the
valid (non-NaN) range -- causal by construction: a value at position t only
ever uses `close[<=t]` for the "signal" side and `close[>t]` for the "forward"
side, mirroring this project's existing causality discipline (see the
detectors/ docstrings and trader/README's "causal by construction" note).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader.config import SessionsConfig
from trader.detectors.structure import detect_swings
from trader.sessions import classify_session

MOMENTUM_REVERSION_N = {
    "M15": (16, 96),
    "H1": (6, 24),
    "D1": (5, 20),
}
Z_WINDOW = 100
Z_THRESHOLD = 2.0
SESSION_HORIZON_BARS = 4  # M15 bars (~1h)


@dataclass(frozen=True)
class DenseFactorResult:
    direction: np.ndarray
    forward_returns: np.ndarray
    n_obs: int


@dataclass(frozen=True)
class SparseFactorResult:
    full_returns: np.ndarray
    event_mask: np.ndarray
    n_events: int


def _log_close(candles: pd.DataFrame) -> np.ndarray:
    return np.log(candles["close"].to_numpy(dtype=float))


def momentum_factor(candles: pd.DataFrame, n: int) -> DenseFactorResult:
    """direction[t] = sign(return over [t-n, t]); forward_returns[t] = return
    over [t, t+n]. Dense/always-on -- use the *_dense_signal permutation test."""
    log_close = _log_close(candles)
    length = len(log_close)
    if length <= 2 * n:
        return DenseFactorResult(np.array([]), np.array([]), 0)

    past_return = log_close[n:] - log_close[:-n]  # index i -> bar (i+n), uses [i, i+n]
    # align: past_return[i] is "as of" bar (i+n). forward_return "as of" bar
    # (i+n) needs log_close[(i+n)+n] - log_close[i+n].
    valid_len = length - 2 * n
    past_at_signal_bar = past_return[:valid_len]  # as of bars n .. n+valid_len-1
    forward = log_close[2 * n : 2 * n + valid_len] - log_close[n : n + valid_len]

    direction = np.sign(past_at_signal_bar)
    keep = direction != 0.0
    return DenseFactorResult(direction[keep], forward[keep], int(keep.sum()))


def mean_reversion_factor(candles: pd.DataFrame, n: int, z_window: int = Z_WINDOW, z_threshold: float = Z_THRESHOLD) -> SparseFactorResult:
    """Event when |rolling z-score of the N-bar return| > threshold. direction
    = -sign(return) (bet against the extreme move). Sparse -- use the
    *_sparse_event permutation test."""
    log_close = _log_close(candles)
    length = len(log_close)
    if length <= 2 * n + z_window:
        return SparseFactorResult(np.array([]), np.array([], dtype=bool), 0)

    past_return = log_close[n:] - log_close[:-n]  # as of bars n..length-1
    s = pd.Series(past_return)
    roll_mean = s.rolling(z_window).mean().to_numpy()
    roll_std = s.rolling(z_window).std(ddof=0).to_numpy()
    z = (past_return - roll_mean) / roll_std

    # forward return "as of" the same signal bars, same alignment as momentum_factor
    signal_len = len(past_return)  # bars n .. length-1, i.e. index i -> bar (i+n)
    valid_len = signal_len - n  # need bar (i+n)+n to exist -> i <= signal_len-1-n
    z = z[:valid_len]
    past_at_signal_bar = past_return[:valid_len]
    forward = log_close[2 * n : 2 * n + valid_len] - log_close[n : n + valid_len]

    valid = ~np.isnan(z) & (roll_std[:valid_len] > 0)
    # trim to the first valid index instead of zero-filling the warm-up region --
    # a filler 0 in `full_returns` would corrupt the permutation-test pool with
    # fake "returns" that never happened, diluting its null distribution.
    first_valid = int(np.argmax(valid)) if valid.any() else valid_len
    direction = -np.sign(past_at_signal_bar[first_valid:])
    forward_trimmed = forward[first_valid:]
    full_returns = forward_trimmed * direction
    event_mask = np.abs(z[first_valid:]) > z_threshold
    return SparseFactorResult(full_returns, event_mask, int(event_mask.sum()))


def compute_session_labels(candles_m15: pd.DataFrame, sessions_config: SessionsConfig) -> pd.DataFrame:
    """One classify_session() pass per bar, shared by all 3 session variants --
    calling it separately per session name would redo the same per-timestamp
    ZoneInfo/date work 3x for no reason."""
    states = candles_m15["timestamp"].apply(lambda ts: classify_session(ts, sessions_config))
    return pd.DataFrame(
        {
            "asia": [s.asia for s in states],
            "london": [s.london for s in states],
            "new_york": [s.new_york for s in states],
        }
    )


def session_factor(candles_m15: pd.DataFrame, session_labels: pd.DataFrame, session_name: str, horizon_bars: int = SESSION_HORIZON_BARS) -> SparseFactorResult:
    """event_mask = bar t falls inside `session_name` (Asia/London/NY, from a
    precomputed `compute_session_labels` frame). full_returns = RAW (unsigned)
    forward log-return over `horizon_bars` -- caller determines the empirical
    sign on EXPLORATION and freezes it."""
    log_close = _log_close(candles_m15)
    length = len(log_close)
    valid_len = length - horizon_bars
    if valid_len <= 0:
        return SparseFactorResult(np.array([]), np.array([], dtype=bool), 0)

    forward = log_close[horizon_bars:horizon_bars + valid_len] - log_close[:valid_len]
    in_session = session_labels[session_name].iloc[:valid_len].to_numpy()
    return SparseFactorResult(forward, in_session, int(in_session.sum()))


def weekday_factor(candles_d1: pd.DataFrame, weekday: int) -> SparseFactorResult:
    """event_mask = bar t's own weekday == `weekday` (0=Monday..4=Friday).
    full_returns = that bar's OWN realized close-to-close log return (not a
    forward-looking prediction -- a seasonality characterization, "is the
    return realized on Mondays different from zero", not a next-bar signal)."""
    log_close = _log_close(candles_d1)
    if len(log_close) < 2:
        return SparseFactorResult(np.array([]), np.array([], dtype=bool), 0)

    realized_return = log_close[1:] - log_close[:-1]  # bar i (1-indexed from orig) return
    timestamps = candles_d1["timestamp"].iloc[1:]
    is_weekday = (timestamps.dt.weekday == weekday).to_numpy()
    return SparseFactorResult(realized_return, is_weekday, int(is_weekday.sum()))


# -- prompt-inventario-real-instrumentos.md: classic-TA battery (swing horizon, D1) --

WARMUP_BARS = 200  # skip unstable EMA/rolling warm-up, same spirit as z_window in mean_reversion_factor


def trend_ma_cross_factor(candles: pd.DataFrame, fast: int, slow: int, horizon: int) -> DenseFactorResult:
    """direction[t] = sign(EMA_fast[t] - EMA_slow[t]); forward_returns[t] =
    return over [t, t+horizon]. Dense/always-on -- use *_dense_signal."""
    close = candles["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean().to_numpy()
    ema_slow = close.ewm(span=slow, adjust=False).mean().to_numpy()
    log_close = _log_close(candles)
    length = len(log_close)

    warmup = max(WARMUP_BARS, slow)
    valid_len = length - warmup - horizon
    if valid_len <= 0:
        return DenseFactorResult(np.array([]), np.array([]), 0)

    direction = np.sign(ema_fast[warmup : warmup + valid_len] - ema_slow[warmup : warmup + valid_len])
    forward = log_close[warmup + horizon : warmup + horizon + valid_len] - log_close[warmup : warmup + valid_len]
    keep = direction != 0.0
    return DenseFactorResult(direction[keep], forward[keep], int(keep.sum()))


def channel_breakout_factor(candles: pd.DataFrame, n: int, horizon: int) -> SparseFactorResult:
    """Donchian-style breakout: close[t] beyond the PRIOR n-bar high/low (the
    channel excludes bar t itself -- `.shift(1)` -- so the breakout isn't
    partly defined by the very bar that triggers it). direction = +1 above the
    prior high, -1 below the prior low. Sparse -- use *_sparse_event."""
    high, low, close = candles["high"], candles["low"], candles["close"]
    prior_high = high.rolling(n).max().shift(1).to_numpy()
    prior_low = low.rolling(n).min().shift(1).to_numpy()
    log_close = _log_close(candles)
    length = len(log_close)

    valid_len = length - horizon
    if valid_len <= 0:
        return SparseFactorResult(np.array([]), np.array([], dtype=bool), 0)

    c = close.to_numpy()[:valid_len]
    ph, pl = prior_high[:valid_len], prior_low[:valid_len]
    valid = ~np.isnan(ph) & ~np.isnan(pl)
    first_valid = int(np.argmax(valid)) if valid.any() else valid_len

    c, ph, pl = c[first_valid:], ph[first_valid:], pl[first_valid:]
    breakout_up = c > ph
    breakout_down = c < pl
    direction = np.where(breakout_up, 1.0, np.where(breakout_down, -1.0, 0.0))
    forward = log_close[first_valid + horizon : first_valid + horizon + len(c)] - log_close[first_valid : first_valid + len(c)]
    event_mask = breakout_up | breakout_down
    full_returns = forward * direction
    return SparseFactorResult(full_returns, event_mask, int(event_mask.sum()))


def volume_spike_factor(candles: pd.DataFrame, multiple: float, horizon: int, avg_window: int = 20) -> SparseFactorResult:
    """Event when bar t's tick volume > `multiple` x the PRIOR `avg_window`-bar
    average (shifted, excludes bar t itself). direction = sign of that bar's
    OWN close-to-close return (volume-confirmed continuation, not a fresh
    prediction of direction). Sparse -- use *_sparse_event.

    Tick volume only -- this account (like every FX/CFD instrument checked in
    the inventory) has no real_volume, confirmed via a raw gateway probe
    (scripts/inventory_symbols.py). A known, weaker proxy than true traded
    volume, not hidden."""
    if "volume" not in candles.columns:
        return SparseFactorResult(np.array([]), np.array([], dtype=bool), 0)

    volume = candles["volume"].to_numpy(dtype=float)
    avg_vol = pd.Series(volume).rolling(avg_window).mean().shift(1).to_numpy()
    log_close = _log_close(candles)
    length = len(log_close)

    valid_len = length - horizon
    if valid_len <= 1:
        return SparseFactorResult(np.array([]), np.array([], dtype=bool), 0)

    own_return = np.empty(valid_len)
    own_return[0] = np.nan
    own_return[1:] = log_close[1:valid_len] - log_close[: valid_len - 1]
    v, av = volume[:valid_len], avg_vol[:valid_len]
    valid = ~np.isnan(av) & ~np.isnan(own_return) & (av > 0)
    first_valid = int(np.argmax(valid)) if valid.any() else valid_len

    v, av, own_return = v[first_valid:], av[first_valid:], own_return[first_valid:]
    event_mask = v > multiple * av
    direction = np.sign(own_return)
    forward = log_close[first_valid + horizon : first_valid + horizon + len(v)] - log_close[first_valid : first_valid + len(v)]
    full_returns = forward * direction
    return SparseFactorResult(full_returns, event_mask & (direction != 0.0), int((event_mask & (direction != 0.0)).sum()))


def fibonacci_reaction_factor(
    candles: pd.DataFrame, timeframe: str, level: float, horizon: int,
    swing_left: int = 2, swing_right: int = 2, tolerance_pct: float = 0.05,
) -> SparseFactorResult:
    """Reuses the existing causal swing detector (trader/detectors/structure.py)
    instead of a parallel reimplementation. A "leg" is 2 consecutive
    alternating-kind swings (low->high = bullish leg, high->low = bearish);
    `level` (e.g. 0.382/0.5/0.618) marks the retracement price INTO that leg.
    "Reaction" = a later bar's low/high (for a bullish/bearish leg) touches
    within `tolerance_pct` of the level AND closes back beyond it the SAME
    bar (immediate rejection -- the simplest unambiguous definition, not
    "eventually" over some window, which would need its own extra parameter
    and could double-count). direction = the leg's own direction (betting the
    pullback holds and the original move resumes). Only the leg most recently
    confirmed as of each bar is considered -- never a stale or future leg."""
    swings = detect_swings(candles, timeframe, swing_left, swing_right)
    ts = candles["timestamp"]  # kept as a tz-aware Series -- .searchsorted() handles tz correctly, np.datetime64() does not
    highs, lows, closes = candles["high"].to_numpy(), candles["low"].to_numpy(), candles["close"].to_numpy()
    n = len(candles)

    full_returns = np.zeros(n)
    event_mask = np.zeros(n, dtype=bool)
    log_close = _log_close(candles)

    # each consecutive alternating-kind pair (swings[i-1], swings[i]) is "the
    # last completed leg" from swings[i]'s own confirmation until the NEXT
    # swing confirms (which starts the following leg) -- never a stale leg
    # bleeding past when a newer one became known, never a future leg used
    # before its second point actually confirmed.
    for i in range(1, len(swings)):
        a, b = swings[i - 1], swings[i]
        if a.kind == b.kind:  # two same-side pivots in a row -- not a leg
            continue
        confirmed_at = int(ts.searchsorted(b.timestamp))
        active_until = int(ts.searchsorted(swings[i + 1].timestamp)) if i + 1 < len(swings) else n

        bullish = a.kind == "swing_low"
        leg_low, leg_high = (a.price, b.price) if bullish else (b.price, a.price)
        leg_range = leg_high - leg_low
        if leg_range <= 0:
            continue
        retr_price = (leg_high - level * leg_range) if bullish else (leg_low + level * leg_range)
        tol = tolerance_pct / 100.0 * retr_price

        for t in range(confirmed_at, min(active_until, n - horizon)):
            if event_mask[t]:
                continue
            if bullish:
                touched = lows[t] <= retr_price + tol
                rejected = closes[t] > retr_price
            else:
                touched = highs[t] >= retr_price - tol
                rejected = closes[t] < retr_price
            if touched and rejected:
                event_mask[t] = True
                direction = 1.0 if bullish else -1.0
                full_returns[t] = direction * (log_close[t + horizon] - log_close[t])

    return SparseFactorResult(full_returns, event_mask, int(event_mask.sum()))


def _rsi(close: pd.Series, period: int) -> np.ndarray:
    """Wilder's RSI, the standard/original formulation (alpha=1/period Wilder
    smoothing, not a plain SMA of gains/losses) -- causal by construction,
    each value only ever uses past deltas."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100.0 - 100.0 / (1.0 + rs)).to_numpy()


def rsi_reversal_factor(candles: pd.DataFrame, period: int, oversold: float, overbought: float, horizon: int) -> SparseFactorResult:
    """Classic RSI mean-reversion rule: RSI[t] < oversold -> bullish reversal
    bet (direction=+1); RSI[t] > overbought -> bearish (direction=-1). Sparse
    -- use *_sparse_event."""
    rsi = _rsi(candles["close"], period)
    log_close = _log_close(candles)
    length = len(log_close)

    valid_len = length - horizon
    if valid_len <= 0:
        return SparseFactorResult(np.array([]), np.array([], dtype=bool), 0)

    r = rsi[:valid_len]
    valid = ~np.isnan(r)
    first_valid = int(np.argmax(valid)) if valid.any() else valid_len

    r = r[first_valid:]
    oversold_event = r < oversold
    overbought_event = r > overbought
    direction = np.where(oversold_event, 1.0, np.where(overbought_event, -1.0, 0.0))
    forward = log_close[first_valid + horizon : first_valid + horizon + len(r)] - log_close[first_valid : first_valid + len(r)]
    event_mask = oversold_event | overbought_event
    full_returns = forward * direction
    return SparseFactorResult(full_returns, event_mask, int(event_mask.sum()))
