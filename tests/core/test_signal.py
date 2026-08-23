from datetime import datetime, timezone

import pytest

from tradingai.core.signal import Direction, TradingSignal


def test_is_actionable():
    signal = TradingSignal(
        symbol="EURUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=Direction.LONG,
        confidence=0.75,
    )
    assert signal.is_actionable(confidence_threshold=0.6)
    assert not signal.is_actionable(confidence_threshold=0.8)


def test_neutral_never_actionable():
    signal = TradingSignal(
        symbol="EURUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=Direction.NEUTRAL,
        confidence=0.99,
    )
    assert not signal.is_actionable(confidence_threshold=0.1)


def test_risk_reward_ratio():
    signal = TradingSignal(
        symbol="EURUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=Direction.LONG,
        confidence=0.8,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
    )
    assert signal.risk_reward_ratio == pytest.approx(2.0)
