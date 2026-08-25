from datetime import datetime, timezone
from unittest.mock import MagicMock

from tradingai.core.signal import Direction, TradingSignal
from tradingai.mt5.order_executor import OrderExecutor
from tradingai.mt5.risk_manager import RiskManager

# Solo lo que OrderExecutor realmente usa para redondeo de volumen -- el sizing por
# riesgo ya NO reconstruye pip/tick a mano (ver `calc_profit`, MT5 lo calcula directo).
SYMBOL_INFO = {
    "volume_step": 0.01,
    "volume_min": 0.01,
    "volume_max": 500.0,
}


def _signal_with_stop(entry: float, sl: float, tp: float, direction: Direction, symbol: str = "EURUSD") -> TradingSignal:
    return TradingSignal(
        symbol=symbol,
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=direction,
        confidence=0.9,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


def _connector(calc_profit: float, calc_margin: float, balance: float = 100000.0) -> MagicMock:
    connector = MagicMock()
    connector.get_symbol_info.return_value = SYMBOL_INFO
    connector.get_symbol_tick.return_value = {"ask": 1.16769, "bid": 1.16759}
    connector.send_order.return_value = {"retcode": 10009}
    connector.calc_profit.return_value = calc_profit
    connector.calc_margin.return_value = calc_margin
    connector.get_account_info.return_value = {
        "balance": balance, "equity": balance, "margin_free": balance,
        "currency": "USD", "leverage": 100,
    }
    return connector


def test_lot_size_uses_real_mt5_profit_calc():
    # $100,000 de balance, 1% de riesgo = $1000. MT5 dice que 1.0 lote pierde $150 si
    # el precio llega al SL (15 pips a $10/pip) -> lote correcto = 1000/150 = 6.67.
    # Margen requerido (2000) muy por debajo del tope (10% de $100k) -> no interfiere.
    connector = _connector(calc_profit=-150.0, calc_margin=2000.0)
    risk_manager = RiskManager(risk_per_trade_pct=1.0, connector=connector)
    signal = _signal_with_stop(entry=1.16769, sl=1.16619, tp=1.17119, direction=Direction.LONG)

    executor = OrderExecutor(connector, risk_manager)
    executor.execute(signal)

    sent_volume = connector.send_order.call_args.args[0]["volume"]
    assert 6.5 < sent_volume < 6.8, f"lote deberia ser ~6.67, salio {sent_volume}"


def test_cfd_instrument_uses_real_profit_not_manual_tick_reconstruction():
    # Regresion del bug real visto en vivo el 2026-08-24 con XAUUSD: el broker opera
    # ese simbolo en modo CFD, donde `trade_tick_value/trade_tick_size` NO representa
    # el $ real por punto de precio (daba un valor 10x menor al real, inflando el lote
    # 10x -- perdida real de -$2580 en una operacion pensada para arriesgar ~$1000).
    # Al calcular el sizing con `calc_profit` (el numero real de MT5, sea cual sea la
    # convencion del instrumento) en vez de reconstruirlo a mano, el resultado es
    # correcto sin importar si el simbolo es forex, CFD o indice.
    # $100 de perdida real por lote (equivalente a $100/lote por cada $1 de movimiento,
    # la economia real de XAUUSD con contrato de 100oz) para una distancia de ~$11.93.
    loss_per_lot = 11.93 * 100  # = 1193.0
    connector = _connector(calc_profit=-loss_per_lot, calc_margin=2000.0)
    risk_manager = RiskManager(risk_per_trade_pct=1.0, connector=connector)
    signal = _signal_with_stop(
        entry=4631.98, sl=4643.91, tp=4609.74, direction=Direction.SHORT, symbol="XAUUSD"
    )

    executor = OrderExecutor(connector, risk_manager)
    executor.execute(signal)

    sent_volume = connector.send_order.call_args.args[0]["volume"]
    # 1000 / 1193 = 0.838 -> el riesgo real debe quedar cerca de 1% del balance, no 10x eso.
    assert 0.7 < sent_volume < 1.0, f"lote deberia ser ~0.84 (riesgo real ~1%), salio {sent_volume}"


def test_tight_stop_loss_lot_size_capped_by_margin():
    # Regresion del bug real visto en vivo el 2026-08-24: con un stop muy ajustado
    # (5.9 pips, igual que el caso real de GBPUSD), el sizing por 1% de riesgo por si
    # solo da ~16.9 lotes -- un tamano que consumio 61% del margen de la cuenta en una
    # sola operacion real y bloqueo el resto de simbolos con "No money". El tope de
    # margen debe reducir el lote, no dejarlo pasar tal cual.
    # $59 de perdida por lote (5.9 pips a $10/pip) -> sizing puro = 1000/59 = 16.95.
    connector = _connector(calc_profit=-59.0, calc_margin=61110.18, balance=100000.0)
    risk_manager = RiskManager(risk_per_trade_pct=1.0, connector=connector)
    signal = _signal_with_stop(entry=1.36312, sl=1.36371, tp=1.36194, direction=Direction.SHORT, symbol="GBPUSD")

    executor = OrderExecutor(connector, risk_manager, max_margin_pct_per_trade=10.0)
    executor.execute(signal)

    sent_volume = connector.send_order.call_args.args[0]["volume"]
    assert sent_volume < 5.0, f"el lote deberia quedar acotado por el tope de margen, salio {sent_volume}"
    assert sent_volume > 0
