"""Envio de ordenes al bridge MT5 a partir de un TradingSignal ya aprobado por el RiskManager."""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from tradingai.core.signal import Direction, TradingSignal
from tradingai.mt5.account import get_account_info
from tradingai.mt5.connector import MT5Connector
from tradingai.mt5.risk_manager import RiskManager
from tradingai.mt5.trade_log import append_trade_event

# Constante estable de la API MT5 (TRADE_RETCODE_DONE); documentada por MetaQuotes,
# se hardcodea aqui porque el lado Linux no tiene el paquete MetaTrader5 disponible.
TRADE_RETCODE_DONE = 10009


class OrderExecutor:
    def __init__(
        self,
        connector: MT5Connector,
        risk_manager: RiskManager,
        magic_number: int = 20260820,
        max_margin_pct_per_trade: float = 10.0,
        trades_log_path: str | None = None,
    ) -> None:
        self.connector = connector
        self.risk_manager = risk_manager
        self.magic_number = magic_number
        self.trades_log_path = trades_log_path
        # Tope de seguridad independiente del sizing por riesgo: `calculate_lot_size`
        # acota cuanto se puede PERDER si el SL salta, pero no cuanto MARGEN consume
        # abrir la posicion. Con un stop muy ajustado (ATR bajo en ese momento), el
        # mismo riesgo en $ puede traducirse en un lote enorme -- visto en vivo el
        # 2026-08-24: GBPUSD con SL a 5.9 pips salio con 16.9 lotes y consumio 61% del
        # margen de la cuenta en una sola posicion, bloqueando el resto de simbolos con
        # "No money". Este tope escala el lote hacia abajo (nunca lo aumenta) si el
        # margen requerido supera este % del balance.
        self.max_margin_pct_per_trade = max_margin_pct_per_trade

    def execute(self, signal: TradingSignal, lot_size: float | None = None) -> dict:
        symbol_info = self.connector.get_symbol_info(signal.symbol)
        order_type = "buy" if signal.direction == Direction.LONG else "sell"

        if lot_size is None:
            # Perdida real para 1.0 lote si el precio llega al SL, calculada por MT5
            # (no reconstruida a mano) -- ver bug real de XAUUSD del 2026-08-24 donde
            # la reconstruccion manual con trade_tick_value/trade_tick_size daba un
            # valor 10x menor al real para un simbolo en modo CFD, inflando el lote.
            loss_per_lot = abs(
                self.connector.calc_profit(signal.symbol, order_type, 1.0, signal.entry_price, signal.stop_loss)
            )
            lot_size = self.risk_manager.calculate_lot_size(signal, loss_per_lot=loss_per_lot)

        # Redondea al step del simbolo y respeta minimo/maximo (evita "Invalid volume").
        volume_step = symbol_info.get("volume_step", 0.01) or 0.01
        lot_size = round(round(lot_size / volume_step) * volume_step, 8)
        lot_size = max(symbol_info.get("volume_min", volume_step), lot_size)
        lot_size = min(symbol_info.get("volume_max", lot_size), lot_size)

        tick = self.connector.get_symbol_tick(signal.symbol)
        price = tick["ask"] if signal.direction == Direction.LONG else tick["bid"]

        required_margin = self.connector.calc_margin(signal.symbol, order_type, lot_size, price)
        account = get_account_info(self.connector)
        max_margin_for_trade = account.balance * (self.max_margin_pct_per_trade / 100)
        if required_margin > max_margin_for_trade > 0:
            scale = max_margin_for_trade / required_margin
            lot_size = round(round((lot_size * scale) / volume_step) * volume_step, 8)
            lot_size = max(symbol_info.get("volume_min", volume_step), lot_size)
            logger.warning(
                f"[{signal.symbol}] Lote reducido de sizing-por-riesgo a "
                f"{lot_size} por tope de margen ({self.max_margin_pct_per_trade}% del balance)"
            )

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
            if self.trades_log_path:
                # En modo hedging el ticket de la posicion nueva coincide con el
                # numero de la orden que la abrio (confirmado en vivo el 2026-08-23).
                append_trade_event(
                    self.trades_log_path,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event="APERTURA",
                    ticket=result.get("order"),
                    symbol=signal.symbol,
                    direction=signal.direction.value,
                    price=price,
                    sl=signal.stop_loss,
                    tp=signal.take_profit,
                    confidence=round(signal.confidence, 4),
                    lot_size=lot_size,
                )

        return result
