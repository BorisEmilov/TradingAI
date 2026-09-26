"""NUEVO -- nada de esto existe todavía en `trader/` (el proyecto anterior
tenía un `RiskManager` con límites de sesión/drawdown/correlación; este
`trader/` reescrito no lo reconstruyó). Deliberadamente independiente del
`SessionsConfig`/`classify_session` de producción (ventanas anchas 8-17 /
killzones 7-10) -- Estrategia 1/2 piden ventanas de ENTRADA propias y más
angostas (08-11 hora local, regla 12), así que se definen acá como
`SessionWindowConfig` nuevos, reutilizando el motor DST-aware existente
(`sessions._in_window`) sin copiarlo.

CORRECCIÓN (Prompt 5, 2026-09-21): la primera versión de este módulo
implementaba "máximo 1 operación por sesión" como un contador GLOBAL
(`sessions_used: set[str]`, sin símbolo) -- un límite entre los 3 símbolos a
la vez. Esa lectura de la regla 14 fue una decisión del prompt anterior, no
de la especificación original, y quedó corregida: la regla real es
concurrencia POR PAR -- no se abre una segunda posición en un símbolo que ya
tiene una abierta (de cualquiera de las 2 estrategias), pero símbolos
distintos SÍ pueden tener posiciones simultáneas. `PositionConcurrencyState`
reemplaza esa parte. La pérdida diaria máxima (-1.5R) SÍ sigue siendo global
a nivel de cuenta -- eso no cambió, y por eso queda en un estado separado
(`DailyLossState`) que ya no se acopla al tracking por símbolo.

1. Ventanas de sesión angostas (Londres/NY 08-11).
2. Concurrencia por símbolo: no hay 2 posiciones abiertas a la vez en el
   mismo par, sin importar la estrategia.
3. Pérdida diaria máxima -1.5R, global, sin excepciones -- estado separado.
4. Tamaño de posición desde % de equity -- replica a propósito la lección ya
   documentada en la memoria del proyecto anterior: `trade_tick_value` es el
   valor de un TICK (`trade_tick_size`), no de un pip completo; convertir mal
   esto dio una vez un lote 10x demasiado grande en cuenta real. Acá se pide
   `trade_tick_value`/`trade_tick_size` explícitos, nunca se asume "tick=pip".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trader.config import SessionWindowConfig
from trader.sessions import _in_window, _window_utc

LONDON_ENTRY_WINDOW = SessionWindowConfig(timezone="Europe/London", start_hour=8, end_hour=11)
NEW_YORK_ENTRY_WINDOW = SessionWindowConfig(timezone="America/New_York", start_hour=8, end_hour=11)

MAX_DAILY_LOSS_R = -1.5
RISK_PCT_MIN, RISK_PCT_MAX = 0.0025, 0.0050  # 0.25% - 0.50% de equity, regla 14 / Estrategia 2 regla 9


def active_entry_session(ts: pd.Timestamp) -> str | None:
    """"london" | "new_york" | None -- None significa fuera de ventana de
    entrada, cualquier estrategia debe esperar (regla 12/4)."""
    if _in_window(ts, LONDON_ENTRY_WINDOW):
        return "london"
    if _in_window(ts, NEW_YORK_ENTRY_WINDOW):
        return "new_york"
    return None


_ENTRY_SESSION_WINDOWS = {"london": LONDON_ENTRY_WINDOW, "new_york": NEW_YORK_ENTRY_WINDOW}


def entry_session_end(session: str, generated_at: pd.Timestamp) -> pd.Timestamp:
    """UTC del cierre (11:00 hora local) de la sesión de entrada -- DST-aware,
    reusa el mismo motor que `active_entry_session` (`sessions._window_utc`).
    Usado para la expiración de la orden límite pendiente (Prompt: ventana de
    expiración = fin de sesión, 2026-09-23, reemplaza el offset fijo de 90
    min) -- NO confundir con la salida temporal post-entrada (`exits.py`,
    sigue fija en 90 min/6 velas, justificación propia y distinta, sin
    cambios). `generated_at` está siempre dentro de la ventana de `session`
    para una señal real (así lo garantiza `active_entry_session` al
    generarla), así que el cierre devuelto siempre es posterior."""
    window = _ENTRY_SESSION_WINDOWS[session]
    _, end_utc = _window_utc(window, generated_at.to_pydatetime())
    return end_utc


# -- concurrencia por símbolo (reemplaza el contador global de sesión) ------

@dataclass
class PositionConcurrencyState:
    open_symbols: set[str] = field(default_factory=set)  # símbolos con una posición abierta AHORA MISMO


def can_open_new_trade(state: PositionConcurrencyState, symbol: str) -> tuple[bool, str | None]:
    """(puede_operar, motivo_de_rechazo_si_no). Nunca muta `state`."""
    if symbol in state.open_symbols:
        return False, f"position_already_open_in_{symbol}"
    return True, None


def record_position_opened(state: PositionConcurrencyState, symbol: str) -> PositionConcurrencyState:
    state.open_symbols.add(symbol)
    return state


def record_position_closed(state: PositionConcurrencyState, symbol: str) -> PositionConcurrencyState:
    state.open_symbols.discard(symbol)
    return state


# -- pérdida diaria máxima, global, separada del tracking por símbolo -------
#
# Confirmado explícitamente (Prompt 6, 2026-09-21): "se termina el día" al
# llegar a -1.5R significa BLOQUEAR ENTRADAS NUEVAS, nunca cerrar posiciones
# ya abiertas de golpe -- esas siguen su curso normal (SL/TP/salida temporal
# ya definidos). Esto no es una interpretación elegida entre dos opciones:
# `DailyLossState` nunca tuvo ni tiene ninguna función que toque
# `PositionConcurrencyState` ni que cierre nada -- por construcción, este
# módulo es estructuralmente incapaz de forzar un cierre. El llamador
# consulta `daily_loss_limit_reached()` ANTES de evaluar una señal nueva
# (bloquea la entrada); una posición que ya estaba abierta cuando se cruzó
# el umbral no se entera de este estado en absoluto -- los dos objetos de
# estado son independientes a propósito. Ver
# `tests/test_mtf_session_risk.py::test_daily_loss_lockout_never_touches_an_already_open_position`.

@dataclass
class DailyLossState:
    date: pd.Timestamp  # normalizado a medianoche UTC, ancla del día en curso
    cumulative_r: float = 0.0
    locked_out: bool = False  # -1.5R alcanzado -- BLOQUEA ENTRADAS NUEVAS, nunca cierra lo ya abierto


def _roll_if_new_day(state: DailyLossState, ts: pd.Timestamp) -> DailyLossState:
    today = ts.normalize()
    if today != state.date:
        return DailyLossState(date=today)
    return state


def daily_loss_limit_reached(state: DailyLossState, ts: pd.Timestamp) -> tuple[DailyLossState, bool]:
    state = _roll_if_new_day(state, ts)
    return state, state.locked_out


def record_daily_result(state: DailyLossState, ts: pd.Timestamp, r_multiple: float) -> DailyLossState:
    state = _roll_if_new_day(state, ts)
    state.cumulative_r += r_multiple
    if state.cumulative_r <= MAX_DAILY_LOSS_R:
        state.locked_out = True
    return state


def compute_position_size_lots(
    equity: float,
    risk_pct: float,
    entry: float,
    sl: float,
    trade_tick_value: float,
    trade_tick_size: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> float:
    """Lotes redondeados HACIA ABAJO al `volume_step` del símbolo (nunca hacia
    arriba -- redondear para arriba arriesgaría más de lo pedido). Si el
    riesgo calculado no alcanza ni `volume_min`, devuelve 0.0 (NO TRADE por
    tamaño, el llamador decide qué hacer -- esta función no fuerza un mínimo
    que arriesgaría más de lo pedido)."""
    if not (RISK_PCT_MIN - 1e-9 <= risk_pct <= RISK_PCT_MAX + 1e-9):
        raise ValueError(f"risk_pct {risk_pct} fuera del rango de la regla 14 [{RISK_PCT_MIN},{RISK_PCT_MAX}]")
    risk_amount = equity * risk_pct
    risk_price_distance = abs(entry - sl)
    if risk_price_distance <= 0:
        return 0.0

    # valor por lote de moverse risk_price_distance = (distancia en TICKS) * valor_por_tick
    ticks = risk_price_distance / trade_tick_size
    value_per_lot = ticks * trade_tick_value
    if value_per_lot <= 0:
        return 0.0

    raw_lots = risk_amount / value_per_lot
    stepped = (raw_lots // volume_step) * volume_step
    stepped = round(stepped, 8)
    if stepped < volume_min:
        return 0.0
    return min(stepped, volume_max)
