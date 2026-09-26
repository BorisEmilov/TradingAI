from trader.detectors.structure import detect_structure_breaks, detect_swings
from trader.events import TF_DURATION
from tests.conftest import build_candles

_H1 = TF_DURATION["H1"]

_BARS = [
    (10.0, 10.2, 9.8, 10.0),
    (10.0, 10.5, 9.9, 10.3),
    (10.3, 11.0, 10.2, 10.8),  # swing high pivot (level 11.0), confirmed next bar
    (10.8, 10.6, 10.0, 10.1),
    (10.1, 10.2, 9.7, 9.9),
    (9.9, 10.0, 9.5, 9.6),  # swing low pivot (level 9.5), confirmed next bar
    (9.6, 9.9, 9.7, 9.8),
    (9.8, 10.6, 9.7, 10.5),
    (10.5, 11.5, 10.4, 11.4),  # close breaks above 11.0 -> BOS bullish
    (11.4, 11.6, 11.0, 11.1),  # swing high pivot (level 11.6), confirmed next bar
    (11.1, 11.3, 9.0, 9.2),  # close breaks below 9.5 -> CHoCH bearish
]


def test_detect_swings_confirms_at_right_offset_not_at_pivot_bar():
    df = build_candles(_BARS, freq="1h")
    swings = detect_swings(df, "H1", left=1, right=1)

    assert [(s.kind, s.price, s.timestamp) for s in swings] == [
        ("swing_high", 11.0, df["timestamp"].iloc[3] + _H1),
        ("swing_low", 9.5, df["timestamp"].iloc[6] + _H1),
        ("swing_high", 11.6, df["timestamp"].iloc[10] + _H1),
    ]


def test_detect_structure_breaks_first_break_is_bos_then_reversal_is_choch():
    df = build_candles(_BARS, freq="1h")
    swings = detect_swings(df, "H1", left=1, right=1)
    breaks = detect_structure_breaks(df, "H1", swings)

    assert len(breaks) == 2

    bos = breaks[0]
    assert bos.kind == "bos"
    assert bos.direction == "bullish"
    assert bos.timestamp == df["timestamp"].iloc[8] + _H1
    assert bos.price == 11.4

    choch = breaks[1]
    assert choch.kind == "choch"
    assert choch.direction == "bearish"
    assert choch.timestamp == df["timestamp"].iloc[10] + _H1
    assert choch.price == 9.2


def test_no_swings_means_no_structure_breaks():
    df = build_candles([(1.0, 1.0, 1.0, 1.0)] * 5, freq="1h")
    breaks = detect_structure_breaks(df, "H1", [])
    assert breaks == []
