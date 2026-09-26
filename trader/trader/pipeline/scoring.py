"""Composite weighted score that replaces the old binary confluence AND-gate,
per `logs/scoring_system_weights_proposal.md` (approved after 2 review
rounds -- see prompt-sistema-puntuacion-ponderada.md and the two
prompt-correcciones-diseno-scoring.md follow-ups).

Weights/tiers below are FIXED CONSTANTS, not config. Only the pass/fail
threshold (`ScoringConfig.min_score_trend`/`min_score_reversal`) is meant to
be tuned/swept -- tying the weights themselves to config would invite exactly
the kind of ad-hoc, n=4-fitted adjustment the design document explicitly
warns against. If a weight ever needs to change, that's a deliberate,
documented edit to this file with fresh ICT justification, not a config
sweep.

Two gates that used to be implicit stay hard gates, unchanged: POI existence
(`no_h1_poi_at_price` in pipeline/engine.py, already an existing gate) and
M15 confirmation existence (`no_m15_confirmation`, already an existing gate
too -- widened to a 3-candle window here, still a hard existence check, not
something this module can score its way around). This module only scores the
QUALITY of an already-guaranteed-to-exist POI and confirmation -- it never
decides whether either exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from trader.events import TF_DURATION, MarketEvent
from trader.sessions import SessionState

if TYPE_CHECKING:
    from trader.pipeline.engine import TimeframeAnalysis

_ZONE_TYPE_POINTS = {
    "order_block_bullish": 12.0,
    "order_block_bearish": 12.0,
    "fair_value_gap_bullish": 9.0,
    "fair_value_gap_bearish": 9.0,
    # Currently dead in practice: pipeline/engine.py's POI candidate pool is
    # order_blocks + fvgs only (inverted zones have no confirmed_at/broken_at
    # eligibility tracking today -- apply_mitigation never runs on them), so
    # a POI passed into this module is never actually one of these two kinds
    # yet. Kept here for completeness with the approved design and so this
    # function scores correctly the day that gap is closed, not because it
    # changes anything today.
    "inverted_order_block_bullish": 6.0,
    "inverted_order_block_bearish": 6.0,
    "inverted_fair_value_gap_bullish": 6.0,
    "inverted_fair_value_gap_bearish": 6.0,
}

_CONFIRMATION_TYPE_POINTS = {
    "turtle_soup_bullish": 20.0,
    "turtle_soup_bearish": 20.0,
    "sharp_turn": 20.0,
    "choch": 14.0,
    "inverted_fair_value_gap_bullish": 9.0,
    "inverted_fair_value_gap_bearish": 9.0,
}

# k = how many M15 candles stale the confirmation is relative to `as_of`.
# k not in this map (k>=3) can't occur in practice -- the existence gate in
# pipeline/engine.py (`window_candles`) already rejects anything that stale
# before this module is ever called.
_FRESHNESS_MULTIPLIER = {0: 1.0, 1: 0.65, 2: 0.35}

_HTF_SWEEP_KINDS = {"liquidity_sweep_bullish", "liquidity_sweep_bearish", "turtle_soup_bullish", "turtle_soup_bearish", "sharp_turn"}


@dataclass(frozen=True)
class ScoreBreakdown:
    bias: float
    poi: float
    confirmation: float
    sweep: float
    session: float
    diversity: float

    @property
    def total(self) -> float:
        return self.bias + self.poi + self.confirmation + self.sweep + self.session + self.diversity

    def as_dict(self) -> dict[str, float]:
        return {
            "bias": self.bias,
            "poi": self.poi,
            "confirmation": self.confirmation,
            "sweep": self.sweep,
            "session": self.session,
            "diversity": self.diversity,
            "total": self.total,
        }


def _d1_bias_score(d1_analysis: "TimeframeAnalysis") -> float:
    if not d1_analysis.structure_events:
        return 0.0
    latest = max(d1_analysis.structure_events, key=lambda e: e.timestamp)

    close_ts = d1_analysis.df["timestamp"] + TF_DURATION["D1"]
    matches = close_ts.index[close_ts == latest.timestamp]
    displacement_score = 0.0
    if len(matches):
        i = matches[0]
        candle_range = float(d1_analysis.df["high"].iloc[i] - d1_analysis.df["low"].iloc[i])
        atr_val = float(d1_analysis.atr.iloc[i]) if i < len(d1_analysis.atr) else float("nan")
        if atr_val and not pd.isna(atr_val) and atr_val > 0:
            ratio = candle_range / atr_val
            if ratio >= 2.5:
                displacement_score = 10.0
            elif ratio >= 1.5:
                displacement_score = 8.0
            elif ratio >= 1.0:
                displacement_score = 5.0
            else:
                displacement_score = 0.0

    elliott_bonus = 0.0
    for e in d1_analysis.elliott:  # 0 or 1 items, see TimeframeAnalysis.from_candles
        if e.direction == latest.direction:
            elliott_bonus = float(e.meta.get("confidence", 0.0)) * 5.0

    return displacement_score + elliott_bonus


def _poi_score(poi: MarketEvent, h1_analysis: "TimeframeAnalysis", as_of: pd.Timestamp) -> float:
    type_score = _ZONE_TYPE_POINTS.get(poi.kind, 0.0)
    freshness_score = 8.0 if poi.mitigated_at is None else 4.0

    confluence_bonus = 0.0
    for e in h1_analysis.support_resistance + h1_analysis.equal_levels:
        if e.timestamp <= as_of and poi.price_low <= e.price <= poi.price_high:
            confluence_bonus = 5.0
            break

    return type_score + freshness_score + confluence_bonus


def _m15_confirmation_score(confirmation: MarketEvent, m15_analysis: "TimeframeAnalysis") -> float:
    if len(m15_analysis.df) == 0:
        return 0.0
    last_ts = m15_analysis.df["timestamp"].iloc[-1] + TF_DURATION["M15"]
    k = round((last_ts - confirmation.timestamp) / TF_DURATION["M15"])
    type_points = _CONFIRMATION_TYPE_POINTS.get(confirmation.kind, 0.0)
    multiplier = _FRESHNESS_MULTIPLIER.get(k, 0.0)
    return type_points * multiplier


def _htf_sweep_bonus(
    direction: str, confirmation_ts: pd.Timestamp, d1_analysis: "TimeframeAnalysis", h1_analysis: "TimeframeAnalysis"
) -> float:
    # D1/H1 only -- the M15 confirmation itself is already scored in its own
    # axis; counting it again here (e.g. when the confirmation IS a turtle
    # soup) would double-count the identical event under two axes.
    for analysis in (d1_analysis, h1_analysis):
        pool = analysis.sweeps + analysis.turtle_soups + analysis.sharp_turns
        for e in pool:
            if e.kind in _HTF_SWEEP_KINDS and e.direction == direction and e.timestamp <= confirmation_ts:
                return 10.0
    return 0.0


def _session_score(session: SessionState) -> float:
    if session.overlap_london_ny or session.killzone_london or session.killzone_new_york:
        return 10.0
    if session.any_active:
        return 5.0
    return 0.0


def _diversity_score(families: set[str], timeframes: set[str]) -> float:
    family_score = min(len(families), 4) * 2.0
    timeframe_bonus = 2.0 if len(timeframes) >= 2 else 0.0
    return family_score + timeframe_bonus


def compute_score(
    direction: str,
    d1_analysis: "TimeframeAnalysis",
    h1_analysis: "TimeframeAnalysis",
    m15_analysis: "TimeframeAnalysis",
    poi: MarketEvent,
    confirmation: MarketEvent,
    session: SessionState,
    confluence_families: set[str],
    confluence_timeframes: set[str],
    as_of: pd.Timestamp,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        bias=_d1_bias_score(d1_analysis),
        poi=_poi_score(poi, h1_analysis, as_of),
        confirmation=_m15_confirmation_score(confirmation, m15_analysis),
        sweep=_htf_sweep_bonus(direction, confirmation.timestamp, d1_analysis, h1_analysis),
        session=_session_score(session),
        diversity=_diversity_score(confluence_families, confluence_timeframes),
    )
