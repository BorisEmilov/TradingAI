from datetime import datetime, timezone

from tradingai.core.signal import Direction, TradingSignal
from tradingai.mt5.risk_manager import RiskManager


def _signal(entry, sl, tp, symbol="EURUSD", direction=Direction.LONG, timestamp=None, confidence=0.8) -> TradingSignal:
    return TradingSignal(
        symbol=symbol,
        timeframe="M15",
        timestamp=timestamp or datetime.now(timezone.utc),
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


def test_rejects_when_max_positions_reached():
    rm = RiskManager(max_open_positions=2, get_open_positions_count=lambda: 2)
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1100))


def test_rejects_above_max_positions_when_no_high_confidence_override_configured():
    # Sin `max_open_positions_high_confidence`, el limite base es absoluto -- ni una
    # confianza altisima lo supera.
    rm = RiskManager(max_open_positions=4, get_open_positions_count=lambda: 4)
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1100, confidence=0.99))


def test_allows_override_above_base_limit_with_high_confidence():
    rm = RiskManager(
        max_open_positions=4, max_open_positions_high_confidence=6, high_confidence_override=0.90,
        get_open_positions_count=lambda: 4,
    )
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200, confidence=0.92))


def test_rejects_override_when_confidence_below_threshold():
    rm = RiskManager(
        max_open_positions=4, max_open_positions_high_confidence=6, high_confidence_override=0.90,
        get_open_positions_count=lambda: 4,
    )
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1100, confidence=0.85))


def test_rejects_when_high_confidence_ceiling_also_reached():
    # El techo alto tampoco es ilimitado -- una vez alcanzado, se rechaza igual
    # aunque la confianza sea altisima.
    rm = RiskManager(
        max_open_positions=4, max_open_positions_high_confidence=6, high_confidence_override=0.90,
        get_open_positions_count=lambda: 6,
    )
    assert not rm.approve(_signal(1.1000, 1.0950, 1.1100, confidence=0.99))


def test_within_base_limit_does_not_need_high_confidence():
    rm = RiskManager(
        max_open_positions=4, max_open_positions_high_confidence=6, high_confidence_override=0.90,
        get_open_positions_count=lambda: 3,
    )
    assert rm.approve(_signal(1.1000, 1.0950, 1.1200, confidence=0.76))


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
    # EURUSD LONG y GBPUSD LONG ya estan ambas "cortas de USD" -- una tercera senal
    # que tambien quede corta de USD (aqui, otra LONG EURUSD) supera el limite de 2.
    # max_positions_per_symbol=None para aislar esta regla de la de "1 por simbolo".
    open_positions = [
        {"symbol": "EURUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
        {"symbol": "GBPUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
    ]
    rm = RiskManager(
        max_correlated_same_direction=2,
        max_positions_per_symbol=None,
        get_open_positions_count=lambda: 2,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="EURUSD", direction=Direction.LONG)
    assert not rm.approve(signal)


def test_rejects_cross_pair_sharing_currency_exposure():
    # EURUSD LONG (+EUR/-USD) y GBPUSD LONG (+GBP/-USD) comparten exposicion a -USD.
    # EURJPY LONG (+EUR/-JPY) no comparte simbolo con ninguna de las dos, pero SI
    # comparte +EUR con EURUSD -- por si sola no basta para bloquear (1 < 2), pero
    # demuestra que el modelo por divisa detecta correlacion entre cruces distintos
    # sin necesitar una lista de grupos fija (gap real que tenia el modelo anterior
    # de "grupos de simbolos exactos", expuesto al anadir pares cruzados el 2026-08-24).
    open_positions = [
        {"symbol": "EURUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
        {"symbol": "EURGBP", "type": "buy", "volume": 1.0, "profit": 0.0},
    ]
    rm = RiskManager(
        max_correlated_same_direction=2,
        get_open_positions_count=lambda: 2,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    # EURJPY LONG tambien queda +EUR -- junto a las 2 posiciones ya +EUR, supera el limite.
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="EURJPY", direction=Direction.LONG)
    assert not rm.approve(signal)


def test_allows_uncorrelated_symbol_regardless_of_existing_exposure():
    open_positions = [
        {"symbol": "EURUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
        {"symbol": "GBPUSD", "type": "buy", "volume": 1.0, "profit": 0.0},
    ]
    rm = RiskManager(
        max_correlated_same_direction=2,
        get_open_positions_count=lambda: 2,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    # AUDCAD no comparte ninguna divisa con EUR/GBP/USD -> sin restriccion de correlacion.
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="AUDCAD", direction=Direction.LONG)
    assert rm.approve(signal)


def test_index_symbol_has_no_currency_correlation():
    # US500 (indice, 5 letras) no descompone en divisas -> nunca se bloquea por esta regla.
    # max_positions_per_symbol=None para aislar esta regla de la de "1 por simbolo".
    open_positions = [{"symbol": "US500", "type": "buy", "volume": 1.0, "profit": 0.0}] * 5
    rm = RiskManager(
        max_correlated_same_direction=2,
        max_positions_per_symbol=None,
        get_open_positions_count=lambda: 5,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    signal = TradingSignal(
        symbol="US500", timeframe="M15", timestamp=datetime.now(timezone.utc),
        direction=Direction.LONG, confidence=0.8, entry_price=100.0, stop_loss=95.0, take_profit=110.0,
    )
    assert rm.approve(signal)


def test_rejects_sl_too_close_to_spread():
    # Piloto en vivo del 2026-08-24: con ATR bajo el SL quedaba a veces a solo 3-5x el
    # spread -- se activaba por ruido/spread en vez de por una reversion real.
    # spread=0.00010, SL a 0.0005 de distancia -> solo 5x el spread, por debajo del piso de 10x.
    rm = RiskManager(
        min_sl_spread_multiple=10.0,
        get_open_positions_count=lambda: 0,
        get_spread=lambda symbol: 0.00010,
    )
    signal = _signal(1.1000, 1.0995, 1.1200)  # SL a 0.0005 (rr>=2.0, aprobado por lo demas)
    assert not rm.approve(signal)


def test_approves_sl_far_enough_from_spread():
    rm = RiskManager(
        min_sl_spread_multiple=10.0,
        get_open_positions_count=lambda: 0,
        get_spread=lambda symbol: 0.00010,
    )
    # SL a 0.0020 de distancia -> 20x el spread, por encima del piso de 10x.
    signal = _signal(1.1000, 1.0980, 1.1400)
    assert rm.approve(signal)


def test_sl_spread_rule_disabled_by_default():
    # Sin min_sl_spread_multiple configurado, la regla no bloquea nada (retrocompatible).
    rm = RiskManager(get_open_positions_count=lambda: 0, get_spread=lambda symbol: 0.00010)
    signal = _signal(1.1000, 1.0995, 1.1100)
    assert rm.approve(signal)


def test_rejects_second_position_on_same_symbol():
    # Regresion del piloto en vivo del 2026-08-24: dos senales SHORT EURGBP casi al
    # mismo precio se ejecutaron seguidas porque nada impedia una segunda posicion en
    # el mismo simbolo -- duplico el riesgo real sobre ese movimiento de precio.
    open_positions = [{"symbol": "EURGBP", "type": "sell", "volume": 3.2, "profit": 0.0}]
    rm = RiskManager(
        max_positions_per_symbol=1,
        max_correlated_same_direction=None,
        get_open_positions_count=lambda: 1,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    signal = _signal(0.8555, 0.8560, 0.8545, symbol="EURGBP", direction=Direction.SHORT)
    assert not rm.approve(signal)


def test_approves_second_position_when_symbol_limit_raised():
    open_positions = [{"symbol": "EURGBP", "type": "sell", "volume": 3.2, "profit": 0.0}]
    rm = RiskManager(
        max_positions_per_symbol=2,
        max_correlated_same_direction=None,
        get_open_positions_count=lambda: 1,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    signal = _signal(0.8555, 0.8560, 0.8545, symbol="EURGBP", direction=Direction.SHORT)
    assert rm.approve(signal)


def test_positions_per_symbol_rule_disabled_when_none():
    open_positions = [{"symbol": "EURGBP", "type": "sell", "volume": 3.2, "profit": 0.0}] * 5
    rm = RiskManager(
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        get_open_positions_count=lambda: 5,
        get_open_positions=lambda: open_positions,
        max_open_positions=10,
    )
    signal = _signal(0.8555, 0.8560, 0.8545, symbol="EURGBP", direction=Direction.SHORT)
    assert rm.approve(signal)


def _open_position_with_sl(symbol="EURUSD", volume=1.0) -> dict:
    return {"symbol": symbol, "type": "sell", "volume": volume, "profit": 0.0, "price_open": 1.1000, "sl": 1.1050}


def test_rejects_when_portfolio_var_exceeds_limit():
    # 2 posiciones ya arriesgan $2000 cada una (segun el mock de perdida real) = $4000.
    # + $1000 de la señal nueva (1% de $100k) = $5000 agregado. Tope de 4% de $100k =
    # $4000 -> $5000 supera el tope, se rechaza.
    open_positions = [_open_position_with_sl(), _open_position_with_sl()]
    rm = RiskManager(
        risk_per_trade_pct=1.0,
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        max_portfolio_risk_pct=4.0,
        get_open_positions_count=lambda: 2,
        get_open_positions=lambda: open_positions,
        get_account_equity=lambda: 100_000.0,
        get_worst_case_loss=lambda *a: 2000.0,
        max_open_positions=10,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="GBPUSD", direction=Direction.LONG)
    assert not rm.approve(signal)


def test_approves_when_portfolio_var_within_limit():
    # Mismo escenario, pero con un tope de 10% de $100k = $10000 -> $5000 esta debajo.
    open_positions = [_open_position_with_sl(), _open_position_with_sl()]
    rm = RiskManager(
        risk_per_trade_pct=1.0,
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        max_portfolio_risk_pct=10.0,
        get_open_positions_count=lambda: 2,
        get_open_positions=lambda: open_positions,
        get_account_equity=lambda: 100_000.0,
        get_worst_case_loss=lambda *a: 2000.0,
        max_open_positions=10,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="GBPUSD", direction=Direction.LONG)
    assert rm.approve(signal)


def test_portfolio_var_ignores_positions_without_sl():
    # Una posicion sin SL puesto (dato ausente) no puede acotarse -> se ignora en vez
    # de reventar o bloquear todo por un dato faltante.
    open_positions = [{"symbol": "EURUSD", "type": "sell", "volume": 1.0, "profit": 0.0, "price_open": None, "sl": None}]
    rm = RiskManager(
        risk_per_trade_pct=1.0,
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        max_portfolio_risk_pct=2.0,
        get_open_positions_count=lambda: 1,
        get_open_positions=lambda: open_positions,
        get_account_equity=lambda: 100_000.0,
        get_worst_case_loss=lambda *a: 2000.0,
        max_open_positions=10,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="GBPUSD", direction=Direction.LONG)
    assert rm.approve(signal)  # solo cuenta el 1% ($1000) de la señal nueva, dentro del 2% ($2000)


def test_rejects_signal_during_news_blackout():
    nfp_time = datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc)  # NFP conocido (ver test_news_calendar.py)
    rm = RiskManager(
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        news_calendar_config={"minutes_before": 15, "minutes_after": 15},
        get_open_positions_count=lambda: 0,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, timestamp=nfp_time)
    assert not rm.approve(signal)


def test_approves_signal_outside_news_blackout():
    far_from_nfp = datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc)
    rm = RiskManager(
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        news_calendar_config={"minutes_before": 15, "minutes_after": 15},
        get_open_positions_count=lambda: 0,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, timestamp=far_from_nfp)
    assert rm.approve(signal)


def test_news_calendar_disabled_by_default():
    nfp_time = datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc)
    rm = RiskManager(
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        get_open_positions_count=lambda: 0,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, timestamp=nfp_time)
    assert rm.approve(signal)


def test_portfolio_var_disabled_by_default():
    open_positions = [_open_position_with_sl()] * 10
    rm = RiskManager(
        max_positions_per_symbol=None,
        max_correlated_same_direction=None,
        get_open_positions_count=lambda: 10,
        get_open_positions=lambda: open_positions,
        get_worst_case_loss=lambda *a: 999999.0,
        max_open_positions=20,
    )
    signal = _signal(1.1000, 1.0950, 1.1200, symbol="GBPUSD", direction=Direction.LONG)
    assert rm.approve(signal)
