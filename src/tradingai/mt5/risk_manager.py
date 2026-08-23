"""Gestion de riesgo: aprueba/rechaza senales y calcula el tamano de posicion.

Reglas: limite de posiciones abiertas, ratio riesgo/beneficio minimo, sizing basado
en % de balance arriesgado por operacion, horario de sesion permitido, drawdown
diario maximo, y limite de posiciones correlacionadas en la misma direccion.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from loguru import logger

from tradingai.core.signal import MIN_RISK_REWARD_RATIO, Direction, TradingSignal
from tradingai.mt5.account import get_account_info
from tradingai.mt5.connector import MT5Connector

_DIRECTION_TO_POSITION_TYPE = {Direction.LONG: "buy", Direction.SHORT: "sell"}


class RiskManager:
    def __init__(
        self,
        risk_per_trade_pct: float = 1.0,
        max_open_positions: int = 3,
        min_risk_reward_ratio: float = MIN_RISK_REWARD_RATIO,
        max_daily_drawdown_pct: float | None = None,
        trading_hours_utc: tuple[int, int] | None = None,
        correlated_groups: list[list[str]] | None = None,
        max_correlated_same_direction: int = 2,
        connector: MT5Connector | None = None,
        get_open_positions_count=None,
        get_open_positions=None,
        get_account_equity=None,
    ) -> None:
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_positions = max_open_positions
        # Piso duro: nunca se acepta un ratio por debajo de MIN_RISK_REWARD_RATIO,
        # aunque se intente configurar mas bajo.
        self.min_risk_reward_ratio = max(min_risk_reward_ratio, MIN_RISK_REWARD_RATIO)
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        # (hora_inicio, hora_fin) en UTC, ej. (7, 21). Si hora_fin < hora_inicio, se
        # interpreta como un rango que cruza medianoche.
        self.trading_hours_utc = trading_hours_utc
        self.correlated_groups = correlated_groups or []
        self.max_correlated_same_direction = max_correlated_same_direction
        self.connector = connector

        # Todas inyectables para poder testear sin conexion real al bridge MT5.
        self._get_open_positions_count = get_open_positions_count or self._default_open_positions_count
        self._get_open_positions = get_open_positions or self._default_open_positions
        self._get_account_equity = get_account_equity or self._default_account_equity

        self._day_start_date: date | None = None
        self._day_start_equity: float | None = None

    def approve(self, signal: TradingSignal) -> bool:
        if self._get_open_positions_count() >= self.max_open_positions:
            logger.debug("Rechazada: limite de posiciones abiertas alcanzado")
            return False

        rr = signal.risk_reward_ratio
        if rr is None or rr < self.min_risk_reward_ratio:
            logger.debug(f"Rechazada: risk/reward insuficiente ({rr})")
            return False

        if self.trading_hours_utc is not None and not self._within_trading_hours(signal):
            logger.debug(f"Rechazada: fuera del horario permitido {self.trading_hours_utc} UTC")
            return False

        if self.max_daily_drawdown_pct is not None and self._daily_drawdown_exceeded():
            logger.debug(f"Rechazada: drawdown diario >= {self.max_daily_drawdown_pct}%")
            return False

        if self.correlated_groups and not self._correlation_ok(signal):
            logger.debug(f"Rechazada: demasiadas posiciones correlacionadas en la misma direccion ({signal.symbol})")
            return False

        return True

    def calculate_lot_size(self, signal: TradingSignal, pip_value_per_lot: float, pip_size: float) -> float:
        """Tamano de posicion segun % de balance arriesgado y distancia al stop loss."""
        if self.connector is None:
            raise RuntimeError("RiskManager necesita un MT5Connector para calcular el tamano de posicion.")

        account = get_account_info(self.connector)
        risk_amount = account.balance * (self.risk_per_trade_pct / 100)

        sl_distance_pips = abs(signal.entry_price - signal.stop_loss) / pip_size
        if sl_distance_pips == 0:
            raise ValueError("Distancia a stop loss invalida (0 pips)")

        lot_size = risk_amount / (sl_distance_pips * pip_value_per_lot)
        return round(lot_size, 2)

    def _within_trading_hours(self, signal: TradingSignal) -> bool:
        start, end = self.trading_hours_utc
        hour = signal.timestamp.astimezone(timezone.utc).hour
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end  # rango que cruza medianoche (ej. 22 -> 6)

    def _daily_drawdown_exceeded(self) -> bool:
        today = datetime.now(timezone.utc).date()
        equity = self._get_account_equity()

        if self._day_start_date != today:
            # Primer chequeo del dia: fija la referencia, no rechaza por drawdown todavia.
            self._day_start_date = today
            self._day_start_equity = equity
            return False

        if not self._day_start_equity:
            return False

        drawdown_pct = (self._day_start_equity - equity) / self._day_start_equity * 100
        return drawdown_pct >= self.max_daily_drawdown_pct

    def _correlation_ok(self, signal: TradingSignal) -> bool:
        position_type = _DIRECTION_TO_POSITION_TYPE.get(signal.direction)
        if position_type is None:
            return True  # senal neutral no deberia llegar aqui, pero por si acaso

        group = next((g for g in self.correlated_groups if signal.symbol in g), None)
        if group is None:
            return True  # simbolo fuera de cualquier grupo definido: sin restriccion

        positions = self._get_open_positions()
        same_direction_count = sum(1 for p in positions if p["symbol"] in group and p["type"] == position_type)
        return same_direction_count < self.max_correlated_same_direction

    def _default_open_positions_count(self) -> int:
        if self.connector is None:
            return 0
        return self.connector.get_open_positions_count()

    def _default_open_positions(self) -> list[dict]:
        if self.connector is None:
            return []
        return self.connector.get_open_positions()

    def _default_account_equity(self) -> float:
        if self.connector is None:
            return 0.0
        return get_account_info(self.connector).equity
