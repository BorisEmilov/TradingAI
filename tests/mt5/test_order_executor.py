from datetime import datetime, timezone
from unittest.mock import MagicMock

from tradingai.core.signal import Direction, TradingSignal
from tradingai.mt5.order_executor import OrderExecutor
from tradingai.mt5.risk_manager import RiskManager

# symbol_info real de EURUSD (broker de 5 digitos): un pip son 10 ticks. Si el
# escalado tick->pip se rompe, el lote sale 10x mas grande de lo debido (bug real
# visto en vivo el 2026-08-23: 66.67 lotes en vez de 6.67, MT5 lo rechazo con
# "No money").
EURUSD_SYMBOL_INFO = {
    "point": 1e-05,
    "trade_tick_size": 1e-05,
    "trade_tick_value": 1.0,
    "volume_step": 0.01,
    "volume_min": 0.01,
    "volume_max": 500.0,
}


def _signal() -> TradingSignal:
    return TradingSignal(
        symbol="EURUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=Direction.LONG,
        confidence=0.9,
        entry_price=1.16769,
        stop_loss=1.16619,  # 15 pips
        take_profit=1.17119,
    )


def test_lot_size_uses_pip_value_not_tick_value():
    connector = MagicMock()
    connector.get_symbol_info.return_value = EURUSD_SYMBOL_INFO
    connector.get_symbol_tick.return_value = {"ask": 1.16769, "bid": 1.16759}
    connector.send_order.return_value = {"retcode": 10009}

    risk_manager = RiskManager(risk_per_trade_pct=1.0, connector=connector)
    # $100,000 de balance, 1% de riesgo = $1000; con SL de 15 pips y $10/pip/lote
    # (pip real = 10 ticks), el lote correcto es 1000 / (15 * 10) = 6.67.
    connector.get_account_info.return_value = {
        "balance": 100000.0, "equity": 100000.0, "margin_free": 100000.0,
        "currency": "USD", "leverage": 100,
    }

    executor = OrderExecutor(connector, risk_manager)
    executor.execute(_signal())

    sent_volume = connector.send_order.call_args.args[0]["volume"]
    assert 6.5 < sent_volume < 6.8, f"lote deberia ser ~6.67, salio {sent_volume} (bug de escala tick/pip)"
