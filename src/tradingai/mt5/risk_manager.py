"""Gestion de riesgo: aprueba/rechaza senales y calcula el tamano de posicion.

Reglas: limite de posiciones abiertas, ratio riesgo/beneficio minimo, sizing basado
en % de balance arriesgado por operacion, horario de sesion permitido, drawdown
diario maximo, y limite de exposicion correlacionada por divisa.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from loguru import logger

from tradingai.core.signal import MIN_RISK_REWARD_RATIO, Direction, TradingSignal
from tradingai.mt5.account import get_account_info
from tradingai.mt5.connector import MT5Connector
from tradingai.mt5.news_calendar import is_news_blackout

def _currency_exposures(symbol: str, direction: Direction) -> list[tuple[str, int]]:
    """Divisas (o pseudo-divisas: XAU/oro, US500/indice) que mueve una posicion, con
    signo (+1 = "comprado" esa divisa, -1 = "vendido"). LONG EURUSD = +EUR/-USD;
    SHORT EURUSD = -EUR/+USD. Symbolo de 6 letras -> par base+quote de 3 letras cada
    una (funciona igual para XAUUSD: +XAU/-USD). Simbolos que no encajan en ese
    patron (ej. US500, un indice) no se descomponen -> sin exposicion derivada.
    """
    if len(symbol) != 6 or direction not in (Direction.LONG, Direction.SHORT):
        return []
    base, quote = symbol[:3], symbol[3:]
    sign = 1 if direction == Direction.LONG else -1
    return [(base, sign), (quote, -sign)]


def _position_direction(position: dict) -> Direction:
    return Direction.LONG if position["type"] == "buy" else Direction.SHORT


class RiskManager:
    def __init__(
        self,
        risk_per_trade_pct: float = 1.0,
        max_open_positions: int = 3,
        max_open_positions_high_confidence: int | None = None,
        high_confidence_override: float = 0.90,
        min_risk_reward_ratio: float = MIN_RISK_REWARD_RATIO,
        max_daily_drawdown_pct: float | None = None,
        trading_hours_utc: tuple[int, int] | None = None,
        max_correlated_same_direction: int | None = 2,
        max_positions_per_symbol: int | None = 1,
        min_sl_spread_multiple: float | None = None,
        max_portfolio_risk_pct: float | None = None,
        news_calendar_config: dict | None = None,
        connector: MT5Connector | None = None,
        get_open_positions_count=None,
        get_open_positions=None,
        get_account_equity=None,
        get_spread=None,
        get_worst_case_loss=None,
    ) -> None:
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_positions = max_open_positions
        # Techo HARD, no ilimitado (2026-08-27, pedido explicito): una señal de
        # confianza muy alta puede superar `max_open_positions`, pero solo hasta
        # este techo -- confianza alta no es lo mismo que "seguro" (el modelo puede
        # estar muy confiado y equivocado, ej. en un cambio de regimen o antes de
        # una noticia), y sin techo un dia con muchos simbolos confiados a la vez
        # (ej. una racha fuerte del dolar) podria acumular demasiada exposicion
        # correlacionada de golpe. None = sin excepcion, `max_open_positions` sigue
        # siendo un techo absoluto. El resto de controles (VaR de portafolio,
        # correlacion por divisa, drawdown) se siguen aplicando igual a estas
        # señales, no se saltan.
        self.max_open_positions_high_confidence = max_open_positions_high_confidence
        self.high_confidence_override = high_confidence_override
        # Piso duro: nunca se acepta un ratio por debajo de MIN_RISK_REWARD_RATIO,
        # aunque se intente configurar mas bajo.
        self.min_risk_reward_ratio = max(min_risk_reward_ratio, MIN_RISK_REWARD_RATIO)
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        # (hora_inicio, hora_fin) en UTC, ej. (7, 21). Si hora_fin < hora_inicio, se
        # interpreta como un rango que cruza medianoche.
        self.trading_hours_utc = trading_hours_utc
        # Limite de posiciones abiertas que comparten exposicion a la MISMA divisa en
        # la MISMA direccion (ver _currency_exposures) -- None = sin limite. Derivado
        # automaticamente del nombre de cada simbolo, no de una lista fija de grupos:
        # con pares cruzados (EURJPY, EURGBP, GBPAUD...) la exposicion compartida no
        # se limita a "el mismo simbolo dos veces", una lista de grupos fija se queda
        # corta en cuanto se anaden cruces (ver piloto del 2026-08-24).
        self.max_correlated_same_direction = max_correlated_same_direction
        # Tope de posiciones abiertas en el MISMO simbolo (sin importar direccion).
        # None = sin limite (permite "piramidar" sobre el mismo par). Anadido tras el
        # piloto en vivo del 2026-08-24: el limite de correlacion por divisa (arriba)
        # tolera hasta `max_correlated_same_direction` posiciones compartiendo una
        # divisa entre simbolos DISTINTOS, pero eso tambien dejaba pasar una segunda
        # posicion en el MISMO simbolo (dos SHORT EURGBP casi al mismo precio) sin
        # que nada lo bloquease -- duplica el riesgo real sobre ese movimiento de
        # precio especifico, algo distinto de una correlacion diversificada entre
        # pares distintos.
        self.max_positions_per_symbol = max_positions_per_symbol
        # Piso de distancia del SL respecto al spread actual (bid-ask), en veces el
        # spread. Anadido tras el piloto en vivo del 2026-08-24: con ATR bajo, el SL
        # calculado por el modelo podia quedar a solo 3-5x el spread -- se activaba por
        # ruido/spread en vez de por una reversion real del precio. None = sin piso.
        self.min_sl_spread_multiple = min_sl_spread_multiple
        # Tope de perdida agregada del PORTAFOLIO ENTERO (todas las posiciones
        # abiertas + la señal nueva) si TODAS tocaran su stop loss a la vez, como %
        # del balance. Los limites anteriores acotan riesgo por operacion/simbolo/
        # divisa, pero ninguno responde a "cuanto podria perder la cuenta si el
        # peor escenario le pega a los 16 simbolos combinados a la vez" -- ese es un
        # riesgo real y distinto (un VaR simplificado). None = sin limite.
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        # Ver tradingai.mt5.news_calendar -- pausa NFP (regla recurrente) y
        # FOMC/CPI (lista manual, sin API externa de pago). None/{} = sin filtro.
        self.news_calendar_config = news_calendar_config
        self.connector = connector

        # Todas inyectables para poder testear sin conexion real al bridge MT5.
        self._get_open_positions_count = get_open_positions_count or self._default_open_positions_count
        self._get_open_positions = get_open_positions or self._default_open_positions
        self._get_account_equity = get_account_equity or self._default_account_equity
        self._get_spread = get_spread or self._default_spread
        self._get_worst_case_loss = get_worst_case_loss or self._default_worst_case_loss

        self._day_start_date: date | None = None
        self._day_start_equity: float | None = None

    def approve(self, signal: TradingSignal) -> bool:
        if not self._within_open_positions_limit(signal):
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

        if self.max_positions_per_symbol is not None and not self._within_positions_per_symbol(signal):
            logger.debug(f"Rechazada: ya hay {self.max_positions_per_symbol} o mas posiciones en {signal.symbol}")
            return False

        if self.max_correlated_same_direction is not None and not self._correlation_ok(signal):
            logger.debug(f"Rechazada: demasiada exposicion correlacionada por divisa ({signal.symbol})")
            return False

        if self.min_sl_spread_multiple is not None and not self._sl_far_enough_from_spread(signal):
            logger.debug(f"Rechazada: SL demasiado cerca del spread actual ({signal.symbol})")
            return False

        if self.max_portfolio_risk_pct is not None and not self._portfolio_var_ok(signal):
            logger.debug(f"Rechazada: riesgo agregado del portafolio superaria {self.max_portfolio_risk_pct}%")
            return False

        if self.news_calendar_config and is_news_blackout(signal.timestamp, self.news_calendar_config):
            logger.debug(f"Rechazada: ventana de bloqueo por calendario economico ({signal.symbol})")
            return False

        return True

    def calculate_lot_size(self, signal: TradingSignal, loss_per_lot: float) -> float:
        """Tamano de posicion segun % de balance arriesgado y la perdida real al SL.

        `loss_per_lot`: perdida en USD para 1.0 lote si el precio llega al stop loss,
        calculada por MT5 (`MT5Connector.calc_profit`) -- respeta la convencion real
        de cada instrumento (forex, CFD, indice, pares con conversion de divisa) en
        vez de reconstruirla a mano con pips/ticks (ver bug real de XAUUSD del
        2026-08-24: esa reconstruccion daba un valor 10x menor al real para un
        simbolo en modo CFD, inflando el lote resultante 10x).
        """
        if self.connector is None:
            raise RuntimeError("RiskManager necesita un MT5Connector para calcular el tamano de posicion.")
        if loss_per_lot <= 0:
            raise ValueError(f"loss_per_lot invalido ({loss_per_lot}); revisa entry_price/stop_loss del signal.")

        account = get_account_info(self.connector)
        risk_amount = account.balance * (self.risk_per_trade_pct / 100)

        lot_size = risk_amount / loss_per_lot
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

    def _within_open_positions_limit(self, signal: TradingSignal) -> bool:
        open_count = self._get_open_positions_count()
        if open_count < self.max_open_positions:
            return True
        # Por encima del limite base: solo pasa si hay un techo alto configurado, la
        # señal tiene confianza suficiente, Y todavia hay lugar bajo ESE techo (nunca
        # sin limite -- ver comentario en __init__).
        if self.max_open_positions_high_confidence is None:
            return False
        if open_count >= self.max_open_positions_high_confidence:
            return False
        return signal.confidence >= self.high_confidence_override

    def _within_positions_per_symbol(self, signal: TradingSignal) -> bool:
        positions = self._get_open_positions()
        same_symbol_count = sum(1 for p in positions if p["symbol"] == signal.symbol)
        return same_symbol_count < self.max_positions_per_symbol

    def _correlation_ok(self, signal: TradingSignal) -> bool:
        new_exposure = dict(_currency_exposures(signal.symbol, signal.direction))
        if not new_exposure:
            return True  # simbolo que no descompone en divisas (ej. un indice): sin restriccion

        positions = self._get_open_positions()
        for currency, sign in new_exposure.items():
            same_direction_count = sum(
                1
                for p in positions
                if dict(_currency_exposures(p["symbol"], _position_direction(p))).get(currency) == sign
            )
            if same_direction_count >= self.max_correlated_same_direction:
                return False
        return True

    def _sl_far_enough_from_spread(self, signal: TradingSignal) -> bool:
        spread = self._get_spread(signal.symbol)
        if spread is None or spread <= 0:
            return True  # sin dato de spread disponible, no bloquear por esta regla
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        return sl_distance >= spread * self.min_sl_spread_multiple

    def _portfolio_var_ok(self, signal: TradingSignal) -> bool:
        account = get_account_info(self.connector) if self.connector else None
        balance = account.balance if account else self._get_account_equity()
        if not balance:
            return True  # sin dato de balance, no bloquear por esta regla

        positions = self._get_open_positions()
        existing_risk = 0.0
        for p in positions:
            sl = p.get("sl")
            price_open = p.get("price_open")
            if not sl or not price_open:
                continue  # posicion sin SL puesto (o dato no disponible): no se puede acotar, se ignora
            existing_risk += self._get_worst_case_loss(p["symbol"], p["type"], p["volume"], price_open, sl)

        # La señal nueva todavia no tiene lote asignado en este punto del flujo
        # (RiskManager.approve() se llama ANTES de calcular el tamano de posicion en
        # OrderExecutor) -- se aproxima su peor perdida por el presupuesto de riesgo
        # nominal (risk_per_trade_pct del balance), que es exactamente lo que
        # `calculate_lot_size` esta diseñado para no superar. El tope de margen
        # posterior solo puede REDUCIR el lote real, nunca aumentarlo, asi que esta
        # aproximacion nunca subestima el riesgo agregado real.
        new_trade_risk = balance * (self.risk_per_trade_pct / 100)

        max_allowed = balance * (self.max_portfolio_risk_pct / 100)
        return (existing_risk + new_trade_risk) <= max_allowed

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

    def _default_spread(self, symbol: str) -> float | None:
        if self.connector is None:
            return None
        tick = self.connector.get_symbol_tick(symbol)
        return tick["ask"] - tick["bid"]

    def _default_worst_case_loss(self, symbol: str, position_type: str, volume: float, price_open: float, sl: float) -> float:
        if self.connector is None:
            return 0.0
        return abs(self.connector.calc_profit(symbol, position_type, volume, price_open, sl))
