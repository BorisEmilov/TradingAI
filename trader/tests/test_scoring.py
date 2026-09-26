import pandas as pd

from trader.events import MarketEvent
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.scoring import (
    _d1_bias_score,
    _diversity_score,
    _htf_sweep_bonus,
    _m15_confirmation_score,
    _poi_score,
    _session_score,
    compute_score,
)
from trader.sessions import SessionState
from tests.conftest import build_candles

_TS = pd.Timestamp("2026-01-05", tz="UTC")


def _empty_analysis(timeframe: str, df=None, **overrides) -> TimeframeAnalysis:
    base = dict(
        timeframe=timeframe, df=df if df is not None else build_candles([], freq="1D"),
        swings=[], structure_events=[], order_blocks=[], inverted_order_blocks=[],
        fvgs=[], inverted_fvgs=[], sweeps=[], equal_levels=[], turtle_soups=[],
        sharp_turns=[], support_resistance=[], elliott=[],
    )
    base.update(overrides)
    return TimeframeAnalysis(**base)


# --- Eje 1: D1 bias strength ---


def test_d1_bias_score_weak_displacement_scores_zero():
    df = build_candles([(10.0, 10.05, 9.98, 10.0)] * 5, freq="1D")  # tiny range candles
    close_ts = df["timestamp"].iloc[-1] + pd.Timedelta(days=1)
    choch = MarketEvent.point("choch", "D1", close_ts, "bullish", 10.0)
    atr = pd.Series([0.5] * 5)  # candle range (~0.07) << ATR -- weak displacement
    analysis = _empty_analysis("D1", df=df, structure_events=[choch], atr=atr)

    assert _d1_bias_score(analysis) == 0.0


def test_d1_bias_score_strong_displacement_scores_ten():
    bars = [(10.0, 10.05, 9.98, 10.0)] * 4 + [(10.0, 11.5, 9.9, 11.4)]  # last candle: big range
    df = build_candles(bars, freq="1D")
    close_ts = df["timestamp"].iloc[-1] + pd.Timedelta(days=1)
    bos = MarketEvent.point("bos", "D1", close_ts, "bullish", 11.4)
    atr = pd.Series([0.5] * 5)  # range=1.6, ratio=3.2x >= 2.5x
    analysis = _empty_analysis("D1", df=df, structure_events=[bos], atr=atr)

    assert _d1_bias_score(analysis) == 10.0


def test_d1_bias_score_elliott_bonus_added_when_same_direction():
    bars = [(10.0, 10.05, 9.98, 10.0)] * 5
    df = build_candles(bars, freq="1D")
    close_ts = df["timestamp"].iloc[-1] + pd.Timedelta(days=1)
    choch = MarketEvent.point("choch", "D1", close_ts, "bullish", 10.0)
    atr = pd.Series([float("nan")] * 5)  # ATR not warmed up -> displacement contributes 0
    elliott = MarketEvent.point("elliott_impulse_context", "D1", close_ts, "bullish", 10.0, confidence=1.0)
    analysis = _empty_analysis("D1", df=df, structure_events=[choch], atr=atr, elliott=[elliott])

    assert _d1_bias_score(analysis) == 5.0  # 0 (displacement) + 1.0*5 (elliott)


def test_d1_bias_score_elliott_bonus_not_added_when_opposite_direction():
    bars = [(10.0, 10.05, 9.98, 10.0)] * 5
    df = build_candles(bars, freq="1D")
    close_ts = df["timestamp"].iloc[-1] + pd.Timedelta(days=1)
    choch = MarketEvent.point("choch", "D1", close_ts, "bullish", 10.0)
    atr = pd.Series([float("nan")] * 5)
    elliott = MarketEvent.point("elliott_impulse_context", "D1", close_ts, "bearish", 10.0, confidence=1.0)
    analysis = _empty_analysis("D1", df=df, structure_events=[choch], atr=atr, elliott=[elliott])

    assert _d1_bias_score(analysis) == 0.0


# --- Eje 2: POI H1 zone quality ---


def test_poi_score_fresh_order_block_no_confluence():
    poi = MarketEvent.zone("order_block_bullish", "H1", _TS, "bullish", price_high=10.0, price_low=9.8)
    analysis = _empty_analysis("H1")

    assert _poi_score(poi, analysis, as_of=_TS) == 20.0  # 12 (type) + 8 (never touched) + 0


def test_poi_score_touched_fvg_with_sr_confluence():
    poi = MarketEvent(
        kind="fair_value_gap_bullish", timeframe="H1", timestamp=_TS, direction="bullish",
        price=9.9, price_high=10.0, price_low=9.8, mitigated=True, mitigated_at=_TS, confirmed_at=_TS,
    )
    sr = MarketEvent.point("support_resistance", "H1", _TS, "neutral", 9.9, touches=3)
    analysis = _empty_analysis("H1", support_resistance=[sr])

    assert _poi_score(poi, analysis, as_of=_TS) == 18.0  # 9 (type) + 4 (touched-held) + 5 (S/R confluence)


def test_poi_score_sr_confluence_ignored_if_in_the_future():
    poi = MarketEvent.zone("order_block_bullish", "H1", _TS, "bullish", price_high=10.0, price_low=9.8)
    future_sr = MarketEvent.point("support_resistance", "H1", _TS + pd.Timedelta(hours=1), "neutral", 9.9, touches=3)
    analysis = _empty_analysis("H1", support_resistance=[future_sr])

    assert _poi_score(poi, analysis, as_of=_TS) == 20.0  # no confluence bonus -- future event not knowable yet


# --- Eje 3: M15 confirmation type x freshness ---


def test_m15_confirmation_score_exact_turtle_soup():
    df = build_candles([(9.8, 9.85, 9.75, 9.8)] * 3, freq="15min")
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
    confirmation = MarketEvent.point("turtle_soup_bullish", "M15", last_close, "bullish", 9.8)
    analysis = _empty_analysis("M15", df=df)

    assert _m15_confirmation_score(confirmation, analysis) == 20.0


def test_m15_confirmation_score_one_candle_stale_choch():
    df = build_candles([(9.8, 9.85, 9.75, 9.8)] * 3, freq="15min")
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
    stale_ts = last_close - pd.Timedelta(minutes=15)  # k=1
    confirmation = MarketEvent.point("choch", "M15", stale_ts, "bullish", 9.8)
    analysis = _empty_analysis("M15", df=df)

    assert _m15_confirmation_score(confirmation, analysis) == 14.0 * 0.65


def test_m15_confirmation_score_two_candles_stale_ifvg():
    df = build_candles([(9.8, 9.85, 9.75, 9.8)] * 3, freq="15min")
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
    stale_ts = last_close - 2 * pd.Timedelta(minutes=15)  # k=2
    confirmation = MarketEvent.point("inverted_fair_value_gap_bearish", "M15", stale_ts, "bearish", 9.8)
    analysis = _empty_analysis("M15", df=df)

    assert _m15_confirmation_score(confirmation, analysis) == 9.0 * 0.35


# --- Eje 4: HTF liquidity sweep bonus ---


def test_htf_sweep_bonus_present_on_h1():
    sweep = MarketEvent.point("liquidity_sweep_bullish", "H1", _TS, "bullish", 9.8)
    d1 = _empty_analysis("D1")
    h1 = _empty_analysis("H1", sweeps=[sweep])

    assert _htf_sweep_bonus("bullish", _TS, d1, h1) == 10.0


def test_htf_sweep_bonus_absent_when_none_matches():
    d1 = _empty_analysis("D1")
    h1 = _empty_analysis("H1")

    assert _htf_sweep_bonus("bullish", _TS, d1, h1) == 0.0


def test_htf_sweep_bonus_ignores_events_after_confirmation():
    sweep = MarketEvent.point("liquidity_sweep_bullish", "H1", _TS + pd.Timedelta(hours=1), "bullish", 9.8)
    d1 = _empty_analysis("D1")
    h1 = _empty_analysis("H1", sweeps=[sweep])

    assert _htf_sweep_bonus("bullish", _TS, d1, h1) == 0.0


# --- Eje 5: session quality ---


def test_session_score_overlap_scores_ten():
    session = SessionState(timestamp=_TS, asia=False, london=True, new_york=True, killzone_london=False, killzone_new_york=False)
    assert _session_score(session) == 10.0


def test_session_score_single_session_scores_five():
    session = SessionState(timestamp=_TS, asia=True, london=False, new_york=False, killzone_london=False, killzone_new_york=False)
    assert _session_score(session) == 5.0


def test_session_score_none_active_scores_zero():
    session = SessionState(timestamp=_TS, asia=False, london=False, new_york=False, killzone_london=False, killzone_new_york=False)
    assert _session_score(session) == 0.0


# --- Eje 6: confluence diversity ---


def test_diversity_score_caps_at_four_families_plus_timeframe_bonus():
    families = {"liquidity", "order_block", "fvg", "structure", "session"}  # 5 families, capped at 4
    timeframes = {"D1", "H1"}
    assert _diversity_score(families, timeframes) == 8.0 + 2.0


def test_diversity_score_single_timeframe_no_bonus():
    families = {"liquidity", "order_block"}
    timeframes = {"H1"}
    assert _diversity_score(families, timeframes) == 4.0 + 0.0


# --- compute_score: full composition ---


def test_compute_score_sums_all_six_axes():
    d1_df = build_candles([(10.0, 10.05, 9.98, 10.0)] * 5, freq="1D")
    d1_close = d1_df["timestamp"].iloc[-1] + pd.Timedelta(days=1)
    d1_bos = MarketEvent.point("bos", "D1", d1_close, "bullish", 10.0)
    d1 = _empty_analysis("D1", df=d1_df, structure_events=[d1_bos], atr=pd.Series([float("nan")] * 5))

    poi = MarketEvent.zone("order_block_bullish", "H1", _TS, "bullish", price_high=10.0, price_low=9.8)
    h1 = _empty_analysis("H1")

    m15_df = build_candles([(9.8, 9.85, 9.75, 9.8)] * 3, freq="15min")
    last_close = m15_df["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
    confirmation = MarketEvent.point("turtle_soup_bullish", "M15", last_close, "bullish", 9.8)
    m15 = _empty_analysis("M15", df=m15_df)

    session = SessionState(timestamp=_TS, asia=False, london=True, new_york=True, killzone_london=False, killzone_new_york=False)

    score = compute_score(
        "bullish", d1, h1, m15, poi, confirmation, session,
        confluence_families={"structure", "order_block"}, confluence_timeframes={"D1", "H1"}, as_of=_TS,
    )

    assert score.bias == 0.0  # weak/unwarmed ATR, no elliott
    assert score.poi == 20.0  # fresh OB
    assert score.confirmation == 20.0  # exact turtle soup
    assert score.sweep == 0.0  # no D1/H1 sweep in this fixture
    assert score.session == 10.0  # overlap
    assert score.diversity == 4.0 + 2.0  # 2 families, 2 timeframes
    assert score.total == 0.0 + 20.0 + 20.0 + 0.0 + 10.0 + 6.0
