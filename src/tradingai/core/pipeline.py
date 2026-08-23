"""Orquestador que conecta las dos mitades del proyecto: IA <-> MT5.

Flujo: datos de mercado (D1/H1/M15/M5) -> features -> prediccion IA -> TradingSignal ->
gestion de riesgo -> ejecucion de ordenes. El ciclo se dispara por cierre de vela M15
(temporalidad de entrada principal), pero cada prediccion usa las 4 temporalidades.
"""

from __future__ import annotations

from loguru import logger

from tradingai.ai.inference.predictor import Predictor
from tradingai.core.signal import TradingSignal
from tradingai.mt5.connector import MT5Connector
from tradingai.mt5.order_executor import OrderExecutor
from tradingai.mt5.risk_manager import RiskManager

# Barras extra por encima de seq_len para absorber el warm-up de indicadores (el mas
# exigente es EMA-200): sin este margen, tras build_feature_pipeline + normalize_ohlcv
# quedarian menos filas utiles que seq_len y el predictor fallaria.
FEATURE_WARMUP_BARS = 250


class TradingPipeline:
    """Une conector MT5, predictor IA, gestor de riesgo y ejecutor de ordenes."""

    def __init__(
        self,
        connector: MT5Connector,
        predictor: Predictor,
        risk_manager: RiskManager,
        executor: OrderExecutor,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.connector = connector
        self.predictor = predictor
        self.risk_manager = risk_manager
        self.executor = executor
        self.confidence_threshold = confidence_threshold

    def run_once(self, symbol: str) -> TradingSignal | None:
        """Ejecuta un ciclo completo para un simbolo: datos -> prediccion -> (posible) orden."""
        candles_by_tf = {
            tf: self.connector.get_candles(symbol, tf, n_candles=seq_len + FEATURE_WARMUP_BARS)
            for tf, seq_len in self.predictor.seq_len_by_tf.items()
        }
        signal = self.predictor.predict(candles_by_tf, symbol=symbol)

        if not signal.is_actionable(self.confidence_threshold):
            logger.debug(
                f"[{symbol}] Senal no operable (direction={signal.direction}, "
                f"confidence={signal.confidence:.2f})"
            )
            return signal

        if not self.risk_manager.approve(signal):
            logger.info(f"[{symbol}] Senal rechazada por gestion de riesgo: {signal}")
            return signal

        logger.info(f"[{symbol}] Ejecutando senal: {signal}")
        self.executor.execute(signal)
        return signal

    def run_loop(self, symbols: list[str]) -> None:
        """Bucle continuo (uso en `scripts/run_live.py`)."""
        for symbol in symbols:
            try:
                self.run_once(symbol)
            except Exception:
                logger.exception(f"Error procesando {symbol}")
