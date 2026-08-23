"""Envio de ordenes al bridge MT5 a partir de un TradingSignal ya aprobado por el RiskManager."""

from __future__ import annotations

from loguru import logger

from tradingai.core.signal import Direction, TradingSignal
from tradingai.mt5.connector import MT5Connector
from tradingai.mt5.risk_manager import RiskManager

# Constante estable de la API MT5 (TRADE_RETCODE_DONE); documentada por MetaQuotes,
# se hardcodea aqui porque el lado Linux no tiene el paquete MetaTrader5 disponible.
TRADE_RETCODE_DONE = 10009


class OrderExecutor:
    def __init__(self, connector: MT5Connector, risk_manager: RiskManager, magic_number: int = 20260820) -> None:
        self.connector = connector
        self.risk_manager = risk_manager
        self.magic_number = magic_number

    def execute(self, signal: TradingSignal, lot_size: float | None = None) -> dict:
        symbol_info = self.connector.get_symbol_info(signal.symbol)
        pip_size = symbol_info["point"] * 10  # aproximacion estandar para pares de 5 digitos

        # trade_tick_value es el valor de UN TICK (trade_tick_size), no de un pip -- en
        # brokers de 5 digitos un pip son 10 ticks. Sin este escalado, el lote sale 10x
        # mas grande de lo debido (visto en vivo: 66.67 lotes en vez de 6.67, rechazado
        # por MT5 con "No money").
        pip_value_per_lot = symbol_info["trade_tick_value"] * (pip_size / symbol_info["trade_tick_size"])

        lot_size = lot_size or self.risk_manager.calculate_lot_size(
            signal, pip_value_per_lot=pip_value_per_lot, pip_size=pip_size
        )

        # Redondea al step del simbolo y respeta minimo/maximo (evita "Invalid volume").
        volume_step = symbol_info.get("volume_step", 0.01) or 0.01
        lot_size = round(round(lot_size / volume_step) * volume_step, 8)
        lot_size = max(symbol_info.get("volume_min", volume_step), lot_size)
        lot_size = min(symbol_info.get("volume_max", lot_size), lot_size)

        tick = self.connector.get_symbol_tick(signal.symbol)
        price = tick["ask"] if signal.direction == Direction.LONG else tick["bid"]

        order_request = {
            "symbol": signal.symbol,
            "volume": lot_size,
            "type": "buy" if signal.direction == Direction.LONG else "sell",
            "price": price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "deviation": 10,
            "magic": self.magic_number,
            "comment": f"tradingai conf={signal.confidence:.2f}",
        }

        result = self.connector.send_order(order_request)
        if result.get("retcode") != TRADE_RETCODE_DONE:
            logger.error(f"Orden rechazada ({result.get('retcode')}): {result.get('comment')}")
        else:
            logger.info(f"Orden ejecutada: {signal.symbol} {signal.direction} lot={lot_size} @ {price}")

        return result
