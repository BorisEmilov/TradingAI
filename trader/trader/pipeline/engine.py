"""Orchestrates the D1 -> H1 -> M15 hierarchy from the strategy spec. H4 was
removed from the system entirely (2026-09-16, prompt-eliminar-gate-d1-h4.md)
after the sensitivity sweep showed D1/H4 alignment was the single largest
source of rejected setups (43%) while contributing no confirmed value beyond
what D1 already establishes -- H4 does not appear anywhere below: not as a
bias filter, not as a confluence source, not as a POI/structure detector.

Two trade CATEGORIES come out of `run_from_analyses()`, tried in order at
every anchor:

  - "trend": direction follows D1 bias, whose latest structure event is a
    BOS (continuation). Needs `scoring.min_score_trend`.
  - "reversal": either (a) D1 bias's latest structure event is a fresh CHoCH
    (the trend just flipped -- trading the new direction is "reversal"
    because it reverses the OLDER established trend, not a stale one), or
    (b) a D1 liquidity sweep against the CURRENT bias is followed by a D1
    turtle soup / sharp turn in that same counter-bias direction (an
    exhaustion signal that hasn't yet flipped the formal structure). Needs
    the stricter `scoring.min_score_reversal`.

Both category minimums are now compared against a composite weighted score
(`trader/pipeline/scoring.py`, see logs/scoring_system_weights_proposal.md),
not a binary confluence-family-count AND-gate -- the old
`confluence.min_confluences`/`min_confluences_reversal` config fields still
exist but are no longer read for gating (2026-09-17,
prompt-sistema-puntuacion-ponderada.md). POI existence and M15 confirmation
existence remain hard gates, evaluated before any score is computed -- see
scoring.py's module docstring for why those two specifically stay binary.

`MultiTimeframePipeline.run_from_analyses()` always executes the stages in
`_STAGE_ORDER` for whichever candidate direction/category it's evaluating,
and returns as soon as one fails -- there is no code path that reaches a
later stage without the earlier ones having been computed first, which is
what "no bypass" means here structurally, not just by convention.

`TimeframeAnalysis.as_of()` is what makes backtesting over years of history
tractable without a second, divergent signal implementation: because every
detector is causal by construction (an event's `.timestamp` is exactly when
it became knowable), running the full detector suite ONCE over the entire
history and then slicing event lists by `timestamp <= cutoff` produces the
IDENTICAL result to recomputing everything from scratch at `cutoff` -- so the
live path (`run()`, truncate-then-recompute) and the backtest replay path
(`from_candles()` once + many cheap `as_of()` calls) are guaranteed to agree,
by construction, not by testing.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, replace

import pandas as pd

from trader.config import TraderConfig
from trader.detectors.common import apply_mitigation
from trader.detectors.elliott import elliott_wave_context
from trader.detectors.fvg import detect_fvg
from trader.detectors.indicators import atr
from trader.detectors.liquidity import (
    detect_equal_levels,
    detect_liquidity_sweeps,
    detect_sharp_turns,
    detect_turtle_soup,
)
from trader.detectors.order_blocks import detect_order_blocks
from trader.detectors.structure import detect_structure_breaks, detect_swings
from trader.detectors.support_resistance import detect_support_resistance
from trader.events import TF_DURATION, MarketEvent, closed_candles_as_of, ensure_utc_sorted
from trader.news_filter import is_high_impact_news_window
from trader.pipeline.confluence import evaluate_confluences
from trader.pipeline.scoring import compute_score
from trader.risk.levels import compute_trade_levels
from trader.sessions import classify_session
from trader.signal import TradingSignal

_STAGE_ORDER = ("D1_bias", "session", "news_filter", "H1_poi", "M15_confirmation", "score", "risk")

_MIN_BARS_PER_TF = 10
_CONFIRMATION_KINDS = {"choch", "turtle_soup_bullish", "turtle_soup_bearish", "sharp_turn"}


@dataclass(frozen=True)
class NoSignal:
    reason: str
    stage: str


def _settled_at(z: MarketEvent) -> pd.Timestamp:
    """The earliest cutoff from which on, re-deriving a zone's
    mitigated_at/broken_at/confirmed_at from cutoff-truncated data gives the
    IDENTICAL result as the full-history values already stored on `z` -- i.e.
    the point past which `as_of()` has nothing left to discover about this
    zone. Used to split zones into "settled" (return as-is, no per-zone work)
    vs "still pending" (needs the real per-cutoff recompute) BEFORE doing any
    of that recompute -- see `as_of()`'s `_zones` for why this matters at
    scale: a zone with `mitigated_at is None` in full history will ALWAYS
    truncate to the identical never-touched state for any cutoff at or after
    its own formation, so settled_at=`z.timestamp`. One with a touch but no
    break settles as soon as the touch itself is visible (settled_at=
    `z.mitigated_at`) since a missing `broken_at` truncates to the same
    "still None" either way. One with both a touch AND a break only settles
    once the break itself is visible (settled_at=`z.broken_at`) -- before
    that, cutoff sees "touched, not yet broken", which computes a DIFFERENT
    confirmed_at than the full-history "touched, eventually broken" case.
    """
    if z.mitigated_at is None:
        return z.timestamp
    if z.broken_at is None:
        return z.mitigated_at
    return z.broken_at


class TimeframeAnalysis:
    """Every detector's output for one timeframe.

    Construct with `from_candles()` (runs the full detector suite). Use
    `as_of(cutoff)` to get a cheap, causally-correct view of an
    already-computed analysis as it would have looked at an earlier point in
    time -- no detector is re-run.
    """

    def __init__(
        self,
        timeframe: str,
        df: pd.DataFrame,
        swings: list[MarketEvent],
        structure_events: list[MarketEvent],
        order_blocks: list[MarketEvent],
        inverted_order_blocks: list[MarketEvent],
        fvgs: list[MarketEvent],
        inverted_fvgs: list[MarketEvent],
        sweeps: list[MarketEvent],
        equal_levels: list[MarketEvent],
        turtle_soups: list[MarketEvent],
        sharp_turns: list[MarketEvent],
        support_resistance: list[MarketEvent],
        elliott: list[MarketEvent],
        atr: pd.Series | None = None,
        grace_period: pd.Timedelta = pd.Timedelta(0),
    ):
        self.timeframe = timeframe
        self.df = df
        self.swings = swings
        self.structure_events = structure_events
        self.order_blocks = order_blocks
        self.inverted_order_blocks = inverted_order_blocks
        self.fvgs = fvgs
        self.inverted_fvgs = inverted_fvgs
        self.sweeps = sweeps
        self.equal_levels = equal_levels
        self.turtle_soups = turtle_soups
        self.sharp_turns = sharp_turns
        self.support_resistance = support_resistance
        self.elliott = elliott
        self.atr = atr if atr is not None else pd.Series(dtype=float)
        self.grace_period = grace_period
        # Lazy cache for as_of()'s settled_at-sorted zone lists, keyed by
        # id(zones_list) so it works for order_blocks AND fvgs without two
        # separate attributes. Only ever populated on an instance that
        # `.as_of()` is actually called on (typically the full-history
        # object in a backtest loop) -- never on the truncated objects
        # as_of() itself returns, since nothing in this codebase chains a
        # second .as_of() call onto an already-truncated result.
        self._settled_sort_cache: dict[int, list[MarketEvent]] = {}
        # Lazy cache for as_of()'s df truncation -- see as_of() for why.
        self._df_close_ts: pd.Series | None = None

    @classmethod
    def from_candles(
        cls, df: pd.DataFrame, timeframe: str, config: TraderConfig, grace_period: pd.Timedelta | None = None
    ) -> "TimeframeAnalysis":
        atr_series = atr(df, period=config.structure.atr_period)

        swings = detect_swings(df, timeframe, config.structure.swing_left_bars, config.structure.swing_right_bars)
        structure_events = detect_structure_breaks(df, timeframe, swings)

        if grace_period is None:
            # Default assumes the invalidation grace period is measured in M15
            # candles regardless of `timeframe` -- true for every production
            # caller (D1/H1/M15 all size their grace period off M15's clock).
            # `grace_period` can be passed explicitly to override this, e.g.
            # prompt-experimento-d1-30m-15m.md's single-LTF pipeline, where the
            # zone being invalidated IS the timeframe being analyzed (M30/M15),
            # not always M15.
            grace_period = TF_DURATION["M15"] * config.zone_lifecycle.invalidation_grace_m15_candles

        raw_obs = detect_order_blocks(
            df, timeframe, structure_events, atr_series, config.structure.displacement_atr_multiple
        )
        order_blocks, inverted_order_blocks = apply_mitigation(raw_obs, df, timeframe, grace_period)

        raw_fvgs = detect_fvg(df, timeframe, config.fvg.min_gap_pct)
        fvgs, inverted_fvgs = apply_mitigation(raw_fvgs, df, timeframe, grace_period)

        sweeps = detect_liquidity_sweeps(df, timeframe, swings, config.liquidity.sweep_wick_min_pct)
        equal_levels = detect_equal_levels(swings, timeframe, config.liquidity.equal_level_tolerance_pct)
        turtle_soups = detect_turtle_soup(
            df, timeframe, config.liquidity.turtle_soup_lookback_bars, config.liquidity.sweep_wick_min_pct
        )
        sharp_turns = detect_sharp_turns(df, timeframe, atr_series, config.structure.displacement_atr_multiple)
        support_resistance = detect_support_resistance(
            swings, timeframe, config.support_resistance.cluster_tolerance_pct, config.support_resistance.min_touches
        )

        elliott_event = elliott_wave_context(df, timeframe, config.elliott.zigzag_deviation_pct)

        def _sorted(events: list[MarketEvent]) -> list[MarketEvent]:
            # Most detectors already emit in timestamp order (single forward pass
            # over the candles), but equal_levels/support_resistance cluster by
            # PRICE first, so their output isn't naturally sorted -- as_of() below
            # needs every list sorted to binary-search it.
            return sorted(events, key=lambda e: e.timestamp)

        return cls(
            timeframe=timeframe,
            df=df,
            swings=_sorted(swings),
            structure_events=_sorted(structure_events),
            order_blocks=_sorted(order_blocks),
            inverted_order_blocks=_sorted(inverted_order_blocks),
            fvgs=_sorted(fvgs),
            inverted_fvgs=_sorted(inverted_fvgs),
            sweeps=_sorted(sweeps),
            equal_levels=_sorted(equal_levels),
            turtle_soups=_sorted(turtle_soups),
            sharp_turns=_sorted(sharp_turns),
            support_resistance=_sorted(support_resistance),
            elliott=[elliott_event] if elliott_event else [],
            atr=atr_series,
            grace_period=grace_period,
        )

    def as_of(self, cutoff: pd.Timestamp) -> "TimeframeAnalysis":
        def _events(events_sorted: list[MarketEvent]) -> list[MarketEvent]:
            # events_sorted is sorted ascending by .timestamp (see from_candles) --
            # binary search the cutoff instead of scanning every event on every
            # call. This matters: a backtest calls as_of() per candidate anchor
            # (thousands of times), each scan previously touched the FULL history's
            # event list regardless of how early cutoff was.
            idx = bisect.bisect_right(events_sorted, cutoff, key=lambda e: e.timestamp)
            return events_sorted[:idx]

        def _zones(zones_sorted: list[MarketEvent]) -> list[MarketEvent]:
            # `mitigated_at`/`broken_at` were found by scanning forward from
            # formation and stopping at the FIRST occurrence of each -- that
            # first occurrence, if it's at or before cutoff, is exactly what a
            # truncated scan would also find (more data after it changes
            # nothing about which candle was first). But if it's AFTER cutoff,
            # naively keeping it would leak knowledge of the future into this
            # snapshot -- as happened here for real: `confirmed_at` computed
            # from the full history can be `None` (broke too fast) purely
            # because of a touch+break that only exists beyond cutoff, which a
            # live/truncated run could not have known about yet. Re-derive
            # `confirmed_at` from the cutoff-truncated touch/break instead of
            # reusing the full-history one.
            #
            # Doing that per-zone recompute for EVERY zone on EVERY as_of()
            # call was, empirically (profiled during
            # prompt-sistema-puntuacion-ponderada.md's threshold sweep, which
            # calls as_of() far more often thanks to the wider M15 window),
            # the dominant cost of the entire backtest -- O(candidates x
            # total_zones_ever_seen), since a late-history cutoff sees nearly
            # every zone the symbol ever produced. Most of those zones need
            # NO work at all: `_settled_at()` identifies the point past which
            # a zone's truncated fields provably match its full-history ones
            # already, so those can be returned as-is via a single bisect
            # instead of an O(k) per-zone recompute. Only the small "still
            # pending as of this cutoff" tail actually needs the real logic.
            cache_key = id(zones_sorted)
            settled_sorted = self._settled_sort_cache.get(cache_key)
            if settled_sorted is None:
                settled_sorted = sorted(zones_sorted, key=_settled_at)
                self._settled_sort_cache[cache_key] = settled_sorted

            split = bisect.bisect_right(settled_sorted, cutoff, key=_settled_at)
            out = settled_sorted[:split]  # settled_at <= cutoff => already correct AND already formed by cutoff
            for z in settled_sorted[split:]:
                if z.timestamp > cutoff:
                    continue  # not formed yet
                touched_at = z.mitigated_at if (z.mitigated_at is not None and z.mitigated_at <= cutoff) else None
                broken_at = z.broken_at if (z.broken_at is not None and z.broken_at <= cutoff) else None
                if touched_at is None:
                    confirmed_at = z.timestamp
                elif broken_at is None or (broken_at - touched_at) >= self.grace_period:
                    confirmed_at = touched_at + self.grace_period
                else:
                    confirmed_at = None
                out.append(
                    z
                    if (touched_at, broken_at, confirmed_at) == (z.mitigated_at, z.broken_at, z.confirmed_at)
                    else replace(
                        z,
                        mitigated=touched_at is not None,
                        mitigated_at=touched_at,
                        broken_at=broken_at,
                        confirmed_at=confirmed_at,
                    )
                )
            # NOT re-sorted by timestamp: nothing downstream of as_of()'s output
            # relies on zone order (every consumer uses max()/comprehensions, see
            # poi_candidates in _evaluate_direction and all_events()) -- the ONLY
            # code that needs timestamp order is `_events()`'s bisect, which only
            # ever runs against `self.swings`/etc on the object `.as_of()` is
            # CALLED ON, never against a previous as_of() result (this codebase
            # never chains as_of() calls). Sorting here was pure waste -- profiled
            # at 25s of the 40s this function took, more expensive than the
            # settled/pending split it was "protecting".
            return out

        # `closed_candles_as_of` re-validates (sortedness, no dupes -- both
        # O(n) scans) AND recomputes close-timestamps for the ENTIRE df on
        # every single call -- wasteful here specifically, since `self.df`
        # is the same never-mutated object for every as_of() call on this
        # instance. Validate once (first call), cache the close-timestamps,
        # then bisect straight to the cutoff row instead of a full boolean
        # mask + `.loc[]` over the whole frame. `closed_candles_as_of`
        # itself is untouched -- this only changes as_of()'s OWN repeated-call
        # pattern, not the general-purpose validated-truncation utility other
        # callers rely on.
        if self._df_close_ts is None:
            ensure_utc_sorted(self.df)
            self._df_close_ts = self.df["timestamp"] + TF_DURATION[self.timeframe]
        cutoff_idx = bisect.bisect_right(self._df_close_ts, cutoff)
        truncated_df = self.df.iloc[:cutoff_idx].reset_index(drop=True)
        return TimeframeAnalysis(
            timeframe=self.timeframe,
            df=truncated_df,
            swings=_events(self.swings),
            structure_events=_events(self.structure_events),
            order_blocks=_zones(self.order_blocks),
            inverted_order_blocks=_events(self.inverted_order_blocks),
            fvgs=_zones(self.fvgs),
            inverted_fvgs=_events(self.inverted_fvgs),
            sweeps=_events(self.sweeps),
            equal_levels=_events(self.equal_levels),
            turtle_soups=_events(self.turtle_soups),
            sharp_turns=_events(self.sharp_turns),
            support_resistance=_events(self.support_resistance),
            elliott=_events(self.elliott),
            # truncated_df is always a PREFIX of self.df (candles are sorted, closed_candles_as_of
            # only ever drops a suffix), so the same prefix slice of atr (index-aligned with self.df)
            # stays causally correct without recomputing it.
            atr=self.atr.iloc[: len(truncated_df)],
            grace_period=self.grace_period,
        )

    def all_events(self) -> list[MarketEvent]:
        return (
            self.structure_events
            + self.order_blocks
            + self.inverted_order_blocks
            + self.fvgs
            + self.inverted_fvgs
            + self.sweeps
            + self.equal_levels
            + self.turtle_soups
            + self.sharp_turns
            + self.support_resistance
            + self.elliott
        )

    def bias(self) -> str | None:
        """Latest established trend from confirmed structure breaks: 'up' | 'down' | None."""
        if not self.structure_events:
            return None
        latest = max(self.structure_events, key=lambda e: e.timestamp)
        return "up" if latest.direction == "bullish" else "down"


class MultiTimeframePipeline:
    def __init__(self, config: TraderConfig):
        self.config = config

    def run(
        self, symbol: str, candles_by_tf: dict[str, pd.DataFrame], as_of: pd.Timestamp, current_price: float
    ) -> TradingSignal | NoSignal:
        analyses = {
            tf: TimeframeAnalysis.from_candles(closed_candles_as_of(candles_by_tf[tf], tf, as_of), tf, self.config)
            for tf in ("D1", "H1", "M15")
        }
        return self.run_from_analyses(symbol, analyses, as_of, current_price)

    def run_from_analyses(
        self, symbol: str, analyses: dict[str, TimeframeAnalysis], as_of: pd.Timestamp, current_price: float
    ) -> TradingSignal | NoSignal:
        d1_analysis, h1_analysis, m15_analysis = analyses["D1"], analyses["H1"], analyses["M15"]

        if min(len(d1_analysis.df), len(h1_analysis.df), len(m15_analysis.df)) < _MIN_BARS_PER_TF:
            return NoSignal(reason="insufficient_history", stage="D1_bias")

        bias = d1_analysis.bias()
        if bias is None:
            return NoSignal(reason="no_d1_bias_established", stage="D1_bias")

        session = classify_session(as_of, self.config.sessions)
        session_ok = (
            (session.overlap_london_ny or session.killzone_london or session.killzone_new_york)
            if self.config.sessions.require_overlap_or_killzone
            else session.any_active
        )
        if not session_ok:
            return NoSignal(reason="outside_active_sessions", stage="session")

        blocked, _news_label = is_high_impact_news_window(as_of, self.config.news_filter)
        if blocked:
            return NoSignal(reason="high_impact_news_window", stage="news_filter")

        bias_direction = "bullish" if bias == "up" else "bearish"
        opposite_direction = "bearish" if bias_direction == "bullish" else "bullish"

        latest_d1_structure = max(d1_analysis.structure_events, key=lambda e: e.timestamp)
        primary_category = "reversal" if latest_d1_structure.kind == "choch" else "trend"
        primary_min_score = (
            self.config.scoring.min_score_reversal
            if primary_category == "reversal"
            else self.config.scoring.min_score_trend
        )
        attempts = [(bias_direction, primary_category, primary_min_score)]

        if self._d1_reversal_trigger(d1_analysis, opposite_direction) is not None:
            attempts.append((opposite_direction, "reversal", self.config.scoring.min_score_reversal))

        last_result: NoSignal | None = None
        for direction, category, min_score in attempts:
            result = self._evaluate_direction(
                symbol, direction, category, min_score, d1_analysis, h1_analysis, m15_analysis,
                session, as_of, current_price, bias,
            )
            if isinstance(result, TradingSignal):
                return result
            last_result = result

        assert last_result is not None
        return last_result

    def _evaluate_direction(
        self,
        symbol: str,
        direction: str,
        category: str,
        min_score: float,
        d1_analysis: TimeframeAnalysis,
        h1_analysis: TimeframeAnalysis,
        m15_analysis: TimeframeAnalysis,
        session,
        as_of: pd.Timestamp,
        current_price: float,
        bias: str,
    ) -> TradingSignal | NoSignal:
        tol = self.config.poi.tolerance_pct / 100.0
        poi_candidates = [
            z
            for z in (h1_analysis.order_blocks + h1_analysis.fvgs)
            if z.direction == direction
            and z.confirmed_at is not None
            and z.confirmed_at <= as_of
            and (z.broken_at is None or as_of < z.broken_at)
            and z.overlaps(current_price * (1 - tol), current_price * (1 + tol))
        ]
        if not poi_candidates:
            return NoSignal(reason="no_h1_poi_at_price", stage="H1_poi")
        poi = max(poi_candidates, key=lambda z: z.timestamp)

        confirmation = self._latest_confirmation(m15_analysis, direction, self.config.m15_confirmation.window_candles)
        if confirmation is None:
            return NoSignal(reason="no_m15_confirmation", stage="M15_confirmation")

        pool = d1_analysis.all_events() + h1_analysis.all_events() + m15_analysis.all_events()
        if session.overlap_london_ny or session.killzone_london or session.killzone_new_york:
            pool = pool + [MarketEvent.point("session", "M15", as_of, direction, current_price)]
        relevant = [e for e in pool if e.direction in (direction, "neutral")]
        # min_confluences=0/min_timeframes=0: no longer a pass/fail AND-gate --
        # families/timeframes are read out below purely to feed the diversity
        # axis of the composite score (and the signal's report), per
        # logs/scoring_system_weights_proposal.md.
        confluences = evaluate_confluences(relevant, min_confluences=0, min_timeframes=0)
        score = compute_score(
            direction, d1_analysis, h1_analysis, m15_analysis, poi, confirmation, session,
            confluences.families, confluences.timeframes, as_of,
        )
        if score.total < min_score:
            return NoSignal(reason="score_below_minimum", stage="score")

        trade_direction = "long" if direction == "bullish" else "short"
        buffer = self._structural_buffer(h1_analysis.df)
        structural_stop = poi.price_low - buffer if trade_direction == "long" else poi.price_high + buffer

        risk_price = (current_price - structural_stop) if trade_direction == "long" else (structural_stop - current_price)
        min_risk = self._minimum_risk_floor(h1_analysis.atr)
        if min_risk is not None and risk_price < min_risk:
            # A structural stop this close to entry is not a valid setup even though it's
            # technically "at a structural point" -- spread alone can consume most or all of
            # the risk. Per the strategy's own rule (never an arbitrary pip-based SL), the fix
            # is to REJECT the setup, not to artificially widen the stop to hit some floor.
            return NoSignal(reason="risk_below_minimum_floor", stage="risk")

        target = self._next_liquidity_target(trade_direction, h1_analysis, current_price)
        if target is None:
            return NoSignal(reason="no_valid_target", stage="risk")

        levels = compute_trade_levels(trade_direction, current_price, structural_stop, target, self.config.risk.min_risk_reward)
        if levels is None:
            return NoSignal(reason="risk_reward_below_minimum", stage="risk")

        return TradingSignal(
            symbol=symbol,
            direction=trade_direction,
            category=category,
            bias_1d=bias,
            session=session,
            poi=poi,
            confirmation=confirmation,
            confluences=confluences,
            score=score,
            levels=levels,
            partial_at_progress_pct=self.config.risk.partial_at_progress_pct,
            generated_at=as_of,
        )

    @staticmethod
    def _d1_reversal_trigger(d1_analysis: TimeframeAnalysis, reversal_direction: str) -> MarketEvent | None:
        """A D1 liquidity sweep against the current bias, followed (same
        timestamp or later) by a D1 turtle soup or sharp turn in that same
        counter-bias direction -- an exhaustion signal that hasn't yet
        flipped the formal D1 structure (that case is the OTHER reversal
        trigger, a fresh CHoCH, handled in run_from_analyses via `bias()`
        itself). Returns the confirming event if the trigger is live, else
        None. No recency window beyond "the most recent sweep" -- documented
        simplification, not re-validated against a later same-direction BOS
        that might have invalidated the reversal attempt.
        """
        sweeps = [e for e in d1_analysis.sweeps if e.direction == reversal_direction]
        if not sweeps:
            return None
        latest_sweep = max(sweeps, key=lambda e: e.timestamp)
        confirmations = [
            e
            for e in (d1_analysis.turtle_soups + d1_analysis.sharp_turns)
            if e.direction == reversal_direction and e.timestamp >= latest_sweep.timestamp
        ]
        if not confirmations:
            return None
        return max(confirmations, key=lambda e: e.timestamp)

    @staticmethod
    def _latest_confirmation(m15_analysis: TimeframeAnalysis, direction: str, window_candles: int) -> MarketEvent | None:
        """A confirmation counts if it fell within the last `window_candles`
        M15 candles (inclusive of the one that just closed). `window_candles=1`
        is the original strict behavior (only the candle that just closed).

        Without SOME limit, the pipeline would happily fire off a stale
        confirmation from days ago just because nothing fresher displaced it
        as "latest" -- harmless when called occasionally live, but silently
        wrong when replayed bar-by-bar for a backtest (every bar after a
        confirmation would keep re-triggering on it). `window_candles>1` is a
        deliberate, bounded relaxation of that rule (see
        prompt-implementar-m15-n2.md), not a removal of it.

        `backtest/engine.py`'s `_candidate_confirmation_timestamps` MUST use
        this same `window_candles` value -- it decides which M15 bars the
        backtest even visits, so if the two windows disagree the backtest
        silently under-counts relative to what this method would actually
        allow live. See test_backtest_engine.py's synchronization test.
        """
        if len(m15_analysis.df) == 0:
            return None
        # event timestamps are always a candle's CLOSE time (see structure.py) --
        # compare against the last candle's close, not its open, or nothing would
        # ever match on M15's own confirmation events. Uses `m15_analysis.timeframe`
        # rather than a literal "M15" -- every production caller passes an M15
        # analysis (identical behavior either way), but
        # prompt-experimento-d1-30m-15m.md's single-LTF pipeline reuses this same
        # method with an M30 analysis, where a hardcoded M15 duration would
        # silently miscompute the freshness window.
        step = TF_DURATION[m15_analysis.timeframe]
        last_ts = m15_analysis.df["timestamp"].iloc[-1] + step
        earliest_ts = last_ts - (window_candles - 1) * step
        candidates = [
            e
            for e in m15_analysis.all_events()
            if earliest_ts <= e.timestamp <= last_ts
            and e.direction == direction
            and (e.kind in _CONFIRMATION_KINDS or e.kind.startswith("inverted_fair_value_gap"))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.timestamp)

    @staticmethod
    def _structural_buffer(h1_df: pd.DataFrame) -> float:
        if len(h1_df) == 0:
            return 0.0001
        recent_range = (h1_df["high"] - h1_df["low"]).tail(20).mean()
        return float(recent_range) * 0.1

    def _minimum_risk_floor(self, h1_atr: pd.Series) -> float | None:
        """`None` (no floor enforced) only when H1 ATR isn't warmed up yet --
        that's an early-history edge case, not grounds to reject every signal
        until enough bars accumulate."""
        if len(h1_atr) == 0:
            return None
        latest = h1_atr.iloc[-1]
        if pd.isna(latest):
            return None
        return self.config.risk.min_risk_atr_multiple * float(latest)

    @staticmethod
    def _next_liquidity_target(trade_direction: str, h1_analysis: TimeframeAnalysis, entry: float) -> float | None:
        candidates: list[float] = []
        candidates.extend(e.price for e in h1_analysis.support_resistance)
        candidates.extend(e.price for e in h1_analysis.equal_levels)
        candidates.extend(s.price for s in h1_analysis.swings)

        if trade_direction == "long":
            above = [p for p in candidates if p > entry]
            return min(above) if above else None
        below = [p for p in candidates if p < entry]
        return max(below) if below else None
