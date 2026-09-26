from trader.detectors.elliott import elliott_wave_context, zigzag_pivots
from tests.conftest import build_candles

_PRICES = [100, 106, 100.5, 108, 101, 112, 104]
_BARS = [(p, p, p, p) for p in _PRICES]


def test_zigzag_pivots_matches_hand_computed_sequence():
    df = build_candles(_BARS, freq="1h")
    pivots = zigzag_pivots(df, "H1", deviation_pct=5.0)

    kinds_and_prices = [(k, p) for _, k, p in pivots]
    assert kinds_and_prices == [
        ("low", 100.0),
        ("high", 106.0),
        ("low", 100.5),
        ("high", 108.0),
        ("low", 101.0),
        ("high", 112.0),
    ]


def test_elliott_wave_context_on_hand_computed_pivots():
    df = build_candles(_BARS, freq="1h")
    event = elliott_wave_context(df, "H1", deviation_pct=5.0)

    assert event is not None
    assert event.kind == "elliott_impulse_context"
    assert event.direction == "bullish"
    assert event.price == 112.0
    assert event.meta["rules_passed"] == 0
    assert event.meta["confidence"] == 0.0


def test_too_few_pivots_returns_none():
    df = build_candles([(100, 100, 100, 100)] * 3, freq="1h")
    assert elliott_wave_context(df, "H1", deviation_pct=5.0) is None
