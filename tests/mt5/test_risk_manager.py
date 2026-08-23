from datetime import datetime, timezone

from tradingai.core.signal import Direction, TradingSignal
from tradingai.mt5.risk_manager import RiskManager


def _signal(entry, sl, tp, symbol="EURUSD", direction=Direction.LONG, timestamp=None) -> TradingSignal:
    return TradingSignal(
        symbol=symbol,
        timeframe="M15",
        timestamp=timestamp or datetime.now(timezone.utc),
        direction=direction,
        confidence=0.8,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


def test_rejects_when_max_positions_reached():
    rm = RiskManager(max_open_positions=2, get_open_positions_count=lambda: 2)
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1100))


def test_rejects_low_risk_reward():
    rm = RiskManager(min_risk_reward_ratio=2.0, get_open_positions_count=lambda: 0)
    # rr = (1.1050-1.1000)/(1.1000-1.0950) = 1.0 < 2.0
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1050))


def test_approves_valid_signal():
    rm = RiskManager(max_open_positions=3, get_open_positions_count=lambda: 0)
    # rr = (1.1200-1.1000)/(1.1000-1.0950) = 4.0, comodamente por encima del piso de 2.0
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200))


def test_min_risk_reward_ratio_has_a_hard_floor():
    # Aunque se pida un ratio mas laxo, nunca se acepta por debajo de MIN_RISK_REWARD_RATIO (2.0).
    rm = RiskManager(min_risk_reward_ratio=1.0, get_open_positions_count=lambda: 0)
    assert rm.min_risk_reward_ratio == 2.0


def test_rejects_outside_trading_hours():
    rm = RiskManager(trading_hours_utc=(7, 21), get_open_positions_count=lambda: 0)
    late_night = datetime(2026, 8, 24, 3, 0, tzinfo=timezone.utc)  # 3am UTC, fuera de 7-21
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1200, timestamp=late_night))


def test_approves_within_trading_hours():
    rm = RiskManager(trading_hours_utc=(7, 21), get_open_positions_count=lambda: 0)
    midday = datetime(2026, 8, 24, 14, 0, tzinfo=timezone.utc)
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200, timestamp=midday))


def test_trading_hours_wraps_past_midnight():
    # Rango 22-6 (sesion asiatica, cruza medianoche).
    rm = RiskManager(trading_hours_utc=(22, 6), get_open_positions_count=lambda: 0)
    at_23 = datetime(2026, 8, 24, 23, 0, tzinfo=timezone.utc)
    at_2 = datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc)
    at_10 = datetime(2026, 8, 24, 10, 0, tzinfo=timezone.utc)
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200, timestamp=at_23))
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200, timestamp=at_2))
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1200, timestamp=at_10))


def test_rejects_after_daily_drawdown_exceeded():
    equities = iter([100_000.0, 94_000.0])  # 6% de caida, por encima del limite de 5%
    rm = RiskManager(
        max_daily_drawdown_pct=5.0,
        get_open_positions_count=lambda: 0,
        get_account_equity=lambda: next(equities),
    )
    # Primer approve() del dia solo fija la referencia (100k), no rechaza.
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200))
    # Segundo approve() ve la equity caida -> rechaza.
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1200))


def test_approves_when_drawdown_within_limit():
    equities = iter([100_000.0, 98_000.0])  # 2% de caida, dentro del limite de 5%
    rm = RiskManager(
        max_daily_drawdown_pct=5.0,
        get_open_positions_count=lambda: 0,
        get_account_equity=lambda: next(equities),
    )
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200))
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200))


def test_rejects_too_many_correlated_same_direction_positions():
    open_positions = [
        {"symbol": "EURUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
        {"symbol": "GBPUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
    ]
    rm = RiskManager(
        correlated_groups=[["EURUSD", "GBPUSD"]],
        max_correlated_same_direction=2,
        get_open_positions_count=lambda: 2,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    # Una tercera posicion LONG en el mismo grupo (EURUSD/GBPUSD) supera el limite de 2.
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="EURUSD", direction=Direction.LONG)
    assert not rm.approve(signal)


def test_allows_uncorrelated_symbol_regardless_of_group_exposure():
    open_positions = [
        {"symbol": "EURUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
        {"symbol": "GBPUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
    ]
    rm = RiskManager(
        correlated_groups=[["EURUSD", "GBPUSD"]],
        max_correlated_same_direction=2,
        get_open_positions_count=lambda: 2,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    # XAUUSD no esta en el grupo EURUSD/GBPUSD -> sin restriccion de correlacion.
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="XAUUSD", direction=Direction.LONG)
    assert rm.approve(signal)
