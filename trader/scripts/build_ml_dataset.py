"""Fase 7 (prompt-fase7-modelo-probabilistico.md), step 1-2: builds the
ELIGIBLE-CANDIDATES dataset -- not the 3-4 rule-system survivors, but every
(symbol, as_of, direction) triple that clears the hard eligibility gates,
labeled with the simulated outcome of actually taking that trade.

HARD GATES (kept, unchanged from every prior phase -- these define "is there
even a candidate here", not "is it a good one"):
  - D1 bias established (prerequisite to compute bias features at all)
  - H1 POI zone valid at current price, in the evaluated direction
  - M15 confirmation event within window_candles=3, in the evaluated direction
  - no high-impact news window active
  - structural SL clears the ATR floor
  - R:R >= min_risk_reward achievable

NOT gates here (session, D1 bias/reversal alignment, confluence diversity,
zone type/freshness, confirmation type/freshness, HTF sweep presence) --
these become FEATURES for the model to learn from, not hand-set thresholds.
Both bullish AND bearish direction are evaluated independently at every M15
candidate anchor -- not just the D1-bias-aligned direction the rule-based
pipeline commits to, since direction-alignment itself is now a feature
(`category`), not a precondition for considering a candidate at all. This is
the "counter_bias_untriggered" category that never existed as a production
candidate before this phase.

CAUSALITY / LOOK-AHEAD: every feature below is computed by filtering the
FULL-HISTORY `TimeframeAnalysis` objects to `event.timestamp <= as_of`
directly (bisect on already-sorted lists, see `_causal` below) rather than
calling the expensive `TimeframeAnalysis.as_of()` -- safe because none of
these features touch the OB/FVG re-mitigation logic `.as_of()` exists for
(a zone's confirmed_at/broken_at are static per-zone facts already; POI/M15
existence only needs a direct timestamp compare against them, exactly the
technique already used in scripts/audit_m15_confirmation.py and
scripts/revalidate_zone_fix.py earlier this project). The one place that
DOES need a causally truncated view (`_structural_buffer`'s "last 20 H1
candles", `_minimum_risk_floor`'s "latest ATR", `_next_liquidity_target`'s
level search) is reproduced here with the SAME semantics as
pipeline/engine.py's originals, fed causally-truncated inputs directly
instead of a full `.as_of()` rebuild. `simulate_trade` itself is called with
the FULL (untruncated) h1_analysis for HTF-invalidation-during-the-trade
checks -- exactly matching how production already calls it (see
backtest/engine.py: `full_analyses["H1"]`, not a truncated view) -- that is
NOT a look-ahead: management logic during a live-held trade legitimately
sees events that happen after entry, up to the bar being simulated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import estimate_symbol_cost
from trader.backtest.engine import _candidate_confirmation_timestamps
from trader.backtest.trade import simulate_trade
from trader.config import load_config
from trader.events import TF_DURATION, MarketEvent
from trader.gateway_client import PythonGetawayClient
from trader.news_filter import is_high_impact_news_window
from trader.pipeline.confluence import FAMILY_BY_KIND
from trader.pipeline.engine import TimeframeAnalysis, _CONFIRMATION_KINDS
from trader.pipeline.scoring import ScoreBreakdown
from trader.risk.levels import compute_trade_levels
from trader.signal import TradingSignal

TIMEFRAMES = ("D1", "H1", "M15")
DEFAULT_COUNTS = {"D1": 1500, "H1": 20000, "M15": 50000}
_ZERO_SCORE = ScoreBreakdown(bias=0.0, poi=0.0, confirmation=0.0, sweep=0.0, session=0.0, diversity=0.0)
_HTF_SWEEP_KINDS = {"liquidity_sweep_bullish", "liquidity_sweep_bearish", "turtle_soup_bullish", "turtle_soup_bearish", "sharp_turn"}


def _causal(events: list[MarketEvent], as_of: pd.Timestamp) -> list[MarketEvent]:
    # Plain linear filter, NOT bisect -- every call site here either passes a
    # single per-category list (individually sorted by `from_candles`, where
    # bisect WOULD be safe) or a CONCATENATION of several such lists (e.g.
    # `all_events()`, `support_resistance + equal_levels`), which is not
    # globally sorted even though each part is. Applying bisect_right to an
    # unsorted concatenation silently returns a garbage split point (found
    # empirically: it picked a "confirmation" from over a year in the future
    # instead of the exact-match one sitting right at `as_of`) -- a real
    # look-ahead-shaped bug, not just a slow path. A plain filter is correct
    # for both cases and, at these list sizes (hundreds to a few thousand),
    # not meaningfully slower than sorting-then-bisecting would be anyway.
    return [e for e in events if e.timestamp <= as_of]


def _is_m15_confirmation(e: MarketEvent) -> bool:
    return e.kind in _CONFIRMATION_KINDS or e.kind.startswith("inverted_fair_value_gap")


def _latest_confirmation_causal(m15_analysis: TimeframeAnalysis, direction: str, as_of: pd.Timestamp, window_candles: int) -> tuple[MarketEvent | None, int]:
    earliest = as_of - (window_candles - 1) * TF_DURATION["M15"]
    candidates = [
        e for e in _causal(m15_analysis.all_events(), as_of)
        if e.timestamp >= earliest and e.direction == direction and _is_m15_confirmation(e)
    ]
    if not candidates:
        return None, -1
    best = max(candidates, key=lambda e: e.timestamp)
    k = round((as_of - best.timestamp) / TF_DURATION["M15"])
    return best, k


def _structural_buffer_from_asof(h1_asof: TimeframeAnalysis) -> float:
    # h1_asof.df is ALREADY causally truncated by .as_of() -- .tail(20) here
    # matches pipeline/engine.py's `_structural_buffer` exactly, just fed a
    # pre-truncated df instead of assuming the caller already truncated it.
    if len(h1_asof.df) == 0:
        return 0.0001
    recent_range = (h1_asof.df["high"] - h1_asof.df["low"]).tail(20).mean()
    return float(recent_range) * 0.1


def _min_risk_floor_from_asof(h1_asof: TimeframeAnalysis, min_risk_atr_multiple: float) -> float | None:
    if len(h1_asof.atr) == 0:
        return None
    latest = h1_asof.atr.iloc[-1]
    if pd.isna(latest):
        return None
    return min_risk_atr_multiple * float(latest)


def _next_liquidity_target_from_asof(trade_direction: str, h1_asof: TimeframeAnalysis, entry: float) -> float | None:
    candidates: list[float] = []
    candidates.extend(e.price for e in h1_asof.support_resistance)
    candidates.extend(e.price for e in h1_asof.equal_levels)
    candidates.extend(e.price for e in h1_asof.swings)
    if trade_direction == "long":
        above = [p for p in candidates if p > entry]
        return min(above) if above else None
    below = [p for p in candidates if p < entry]
    return max(below) if below else None


def _d1_reversal_trigger_causal(d1_analysis: TimeframeAnalysis, reversal_direction: str, as_of: pd.Timestamp) -> MarketEvent | None:
    sweeps = [e for e in _causal(d1_analysis.sweeps, as_of) if e.direction == reversal_direction]
    if not sweeps:
        return None
    latest_sweep = max(sweeps, key=lambda e: e.timestamp)
    confirmations = [
        e for e in _causal(d1_analysis.turtle_soups + d1_analysis.sharp_turns, as_of)
        if e.direction == reversal_direction and e.timestamp >= latest_sweep.timestamp
    ]
    if not confirmations:
        return None
    return max(confirmations, key=lambda e: e.timestamp)


def _d1_bias_features(d1_analysis: TimeframeAnalysis, latest_d1: MarketEvent, as_of: pd.Timestamp) -> tuple[float, float]:
    close_ts = d1_analysis.df["timestamp"] + TF_DURATION["D1"]
    matches = close_ts.index[close_ts == latest_d1.timestamp]
    displacement_ratio = float("nan")
    if len(matches):
        i = matches[0]
        candle_range = float(d1_analysis.df["high"].iloc[i] - d1_analysis.df["low"].iloc[i])
        atr_val = float(d1_analysis.atr.iloc[i]) if i < len(d1_analysis.atr) else float("nan")
        if atr_val and not pd.isna(atr_val) and atr_val > 0:
            displacement_ratio = candle_range / atr_val
    elliott_confidence = 0.0
    for e in _causal(d1_analysis.elliott, as_of):
        if e.direction == latest_d1.direction:
            elliott_confidence = float(e.meta.get("confidence", 0.0))
    return displacement_ratio, elliott_confidence


def _poi_features(poi: MarketEvent, h1_asof: TimeframeAnalysis) -> tuple[bool, bool]:
    fresh = poi.mitigated_at is None
    sr_confluence = False
    for e in h1_asof.support_resistance + h1_asof.equal_levels:
        if poi.price_low <= e.price <= poi.price_high:
            sr_confluence = True
            break
    return fresh, sr_confluence


def _htf_sweep_present(direction: str, confirmation_ts: pd.Timestamp, d1_analysis: TimeframeAnalysis, h1_asof: TimeframeAnalysis) -> bool:
    # D1's sweep/turtle_soup/sharp_turn are POINT events (no mitigated_at/
    # confirmed_at/broken_at ever set on them) -- a plain `<=` filter on the
    # full-history D1 analysis is causally safe, unlike zones. H1's are
    # already-causal via h1_asof (needed if this ever changes).
    for e in _causal(d1_analysis.sweeps + d1_analysis.turtle_soups + d1_analysis.sharp_turns, confirmation_ts):
        if e.kind in _HTF_SWEEP_KINDS and e.direction == direction:
            return True
    for e in h1_asof.sweeps + h1_asof.turtle_soups + h1_asof.sharp_turns:
        if e.kind in _HTF_SWEEP_KINDS and e.direction == direction and e.timestamp <= confirmation_ts:
            return True
    return False


def _session_label(as_of: pd.Timestamp, config) -> tuple[str, dict]:
    from trader.sessions import classify_session

    session = classify_session(as_of, config.sessions)
    if session.overlap_london_ny or session.killzone_london or session.killzone_new_york:
        label = "overlap_or_killzone"
    elif session.any_active:
        label = "single_session"
    else:
        label = "none"
    return label, session


def _diversity_features(direction: str, d1_analysis, h1_analysis, m15_analysis, as_of: pd.Timestamp, session_state, current_price: float) -> tuple[int, int]:
    pool = (
        _causal(d1_analysis.all_events(), as_of)
        + _causal(h1_analysis.all_events(), as_of)
        + _causal(m15_analysis.all_events(), as_of)
    )
    if session_state.overlap_london_ny or session_state.killzone_london or session_state.killzone_new_york:
        pool = pool + [MarketEvent.point("session", "M15", as_of, direction, current_price)]
    relevant = [e for e in pool if e.direction in (direction, "neutral")]
    families = {FAMILY_BY_KIND[e.kind] for e in relevant if e.kind in FAMILY_BY_KIND}
    timeframes = {e.timeframe for e in relevant}
    return len(families), len(timeframes)


def _build_signal_for_simulation(symbol, direction, category, session_state, poi, confirmation, levels, as_of) -> TradingSignal:
    from trader.pipeline.confluence import ConfluenceCheck

    return TradingSignal(
        symbol=symbol, direction=levels.direction, category=category, bias_1d=("up" if direction == "bullish" else "down"),
        session=session_state, poi=poi, confirmation=confirmation,
        confluences=ConfluenceCheck(passed=True, families=set(), timeframes=set(), events=[]),
        score=_ZERO_SCORE, levels=levels, partial_at_progress_pct=0.5, generated_at=as_of,
    )


def build_dataset_for_symbol(symbol: str, config, client: PythonGetawayClient) -> list[dict]:
    candles = {tf: client.candles(symbol, tf, DEFAULT_COUNTS[tf], timeout=120) for tf in TIMEFRAMES}
    info = client.symbol_info(symbol, timeout=30)
    cost = estimate_symbol_cost(symbol, float(info["point"]), candles["M15"])

    d1_analysis = TimeframeAnalysis.from_candles(candles["D1"], "D1", config)
    h1_analysis = TimeframeAnalysis.from_candles(candles["H1"], "H1", config)
    m15_analysis = TimeframeAnalysis.from_candles(candles["M15"], "M15", config)
    m15_df = candles["M15"]

    tol = config.poi.tolerance_pct / 100.0
    window_candles = config.m15_confirmation.window_candles
    min_rr = config.risk.min_risk_reward

    m15_close_ts = m15_df["timestamp"] + TF_DURATION["M15"]
    idx_by_m15_close = {t: i for i, t in enumerate(m15_close_ts)}

    anchors = _candidate_confirmation_timestamps(m15_analysis, window_candles)
    print(f"  {symbol}: {len(anchors)} anchors x 2 directions = {2 * len(anchors)} raw candidate-direction pairs")

    rows: list[dict] = []
    n_eligible = 0
    for as_of in anchors:
        bar_idx = idx_by_m15_close.get(as_of)
        if bar_idx is None:
            continue
        current_price = float(m15_df["close"].iloc[bar_idx])

        d1_events_so_far = [e for e in d1_analysis.structure_events if e.timestamp <= as_of]
        if not d1_events_so_far:
            continue
        latest_d1 = max(d1_events_so_far, key=lambda e: e.timestamp)
        bias_direction = latest_d1.direction

        blocked, _ = is_high_impact_news_window(as_of, config.news_filter)
        if blocked:
            continue

        # Zones (order_blocks/fvgs) need PROPERLY re-derived confirmed_at/
        # mitigated_at/broken_at for this specific as_of -- their full-history
        # values reflect what happens over the ENTIRE dataset, not what was
        # knowable at this cutoff (a zone touched-then-settled AFTER as_of
        # would wrongly look "not yet eligible" if you naively used its
        # full-history confirmed_at, when causally it was still fresh at this
        # point). This is exactly what TimeframeAnalysis.as_of() exists for
        # -- reuse it (already validated, already optimized) rather than
        # re-deriving zone settling by hand a second time. Point-event lists
        # (structure/sweeps/swings/S-R/elliott/etc.) don't have this issue --
        # their `.timestamp` is a static formation/confirmation fact, safe to
        # filter with a plain `<=` compare (see `_causal`).
        h1_asof = h1_analysis.as_of(as_of)

        for direction in ("bullish", "bearish"):
            poi_candidates = [
                z for z in (h1_asof.order_blocks + h1_asof.fvgs)
                if z.direction == direction and z.confirmed_at is not None and z.confirmed_at <= as_of
                and (z.broken_at is None or as_of < z.broken_at)
                and z.overlaps(current_price * (1 - tol), current_price * (1 + tol))
            ]
            if not poi_candidates:
                continue
            poi = max(poi_candidates, key=lambda z: z.timestamp)

            confirmation, k = _latest_confirmation_causal(m15_analysis, direction, as_of, window_candles)
            if confirmation is None:
                continue

            trade_direction = "long" if direction == "bullish" else "short"
            buffer = _structural_buffer_from_asof(h1_asof)
            structural_stop = poi.price_low - buffer if trade_direction == "long" else poi.price_high + buffer

            risk_price = (current_price - structural_stop) if trade_direction == "long" else (structural_stop - current_price)
            min_risk = _min_risk_floor_from_asof(h1_asof, config.risk.min_risk_atr_multiple)
            if min_risk is not None and risk_price < min_risk:
                continue

            target = _next_liquidity_target_from_asof(trade_direction, h1_asof, current_price)
            if target is None:
                continue

            levels = compute_trade_levels(trade_direction, current_price, structural_stop, target, min_rr)
            if levels is None:
                continue

            # ELIGIBLE -- extract features and simulate.
            n_eligible += 1

            if direction == bias_direction and latest_d1.kind == "bos":
                category = "trend"
            elif direction == bias_direction and latest_d1.kind == "choch":
                category = "reversal_aligned"
            else:
                opposite = "bearish" if bias_direction == "bullish" else "bullish"
                if direction == opposite and _d1_reversal_trigger_causal(d1_analysis, opposite, as_of) is not None:
                    category = "reversal_triggered"
                else:
                    category = "counter_bias_untriggered"

            displacement_ratio, elliott_confidence = _d1_bias_features(d1_analysis, latest_d1, as_of)
            poi_fresh, poi_sr_confluence = _poi_features(poi, h1_asof)
            htf_sweep = _htf_sweep_present(direction, confirmation.timestamp, d1_analysis, h1_asof)
            session_label, session_state = _session_label(as_of, config)
            n_families, n_timeframes = _diversity_features(direction, d1_analysis, h1_analysis, m15_analysis, as_of, session_state, current_price)

            signal = _build_signal_for_simulation(symbol, direction, category, session_state, poi, confirmation, levels, as_of)
            # simulate_trade gets the FULL (untruncated) h1_analysis, exactly like
            # production (backtest/engine.py passes `full_analyses["H1"]`) -- HTF
            # invalidation-during-the-trade legitimately needs to see structure
            # events that happen AFTER entry, up to the bar being simulated. Not
            # a look-ahead: this is real-time management logic, not signal generation.
            trade = simulate_trade(signal, m15_df, bar_idx, h1_analysis, cost, FAMILY_BY_KIND.get(confirmation.kind, "unknown"))
            if trade is None:
                continue  # gap between signal and fill flipped the setup invalid

            rows.append(
                {
                    "symbol": symbol,
                    "as_of": as_of.isoformat(),
                    "direction": direction,
                    "category": category,
                    "d1_bias_matches_direction": direction == bias_direction,
                    "d1_structure_kind": latest_d1.kind,
                    "d1_displacement_ratio": displacement_ratio,
                    "d1_elliott_confidence": elliott_confidence,
                    "poi_kind": poi.kind,
                    "poi_fresh": poi_fresh,
                    "poi_sr_confluence": poi_sr_confluence,
                    "confirmation_kind": confirmation.kind,
                    "confirmation_staleness_k": k,
                    "htf_sweep_present": htf_sweep,
                    "session_label": session_label,
                    "n_confluence_families": n_families,
                    "n_confluence_timeframes": n_timeframes,
                    "planned_rr": levels.risk_reward,
                    "net_r": trade.net_r,
                    "win": trade.net_r > 0,
                    "exit_reason": trade.exit_reason,
                }
            )

    print(f"  {symbol}: {n_eligible} elegibles, {len(rows)} simulados con exito (n_eligible-simulados se pierden por gap de fill)")
    return rows


def main() -> None:
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()

    all_rows: list[dict] = []
    try:
        for symbol in config.symbols:
            print(f"=== {symbol} ===")
            all_rows.extend(build_dataset_for_symbol(symbol, config, client))
    finally:
        client.logout()

    logs_dir = Path(config.signals_log_path).parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "ml_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")

    print(f"\nTotal filas en el dataset: {len(all_rows)}")
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
