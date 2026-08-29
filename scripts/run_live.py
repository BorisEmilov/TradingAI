"""Bucle en vivo: conecta al bridge MT5, espera cierre de vela, predice y (si aplica) ejecuta.

Uso:
    python scripts/run_live.py --checkpoint data/models/EURUSD_transformer.pt --symbol EURUSD

Requiere el bridge corriendo (scripts/start_mt5_bridge.sh dentro de Wine).
Ver README > "MT5 en Linux".
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.multi_timeframe import ANCHOR_TIMEFRAME  # noqa: E402
from tradingai.ai.inference.gbm_predictor import GBMPredictor  # noqa: E402
from tradingai.ai.inference.predictor import Predictor  # noqa: E402
from tradingai.core.pipeline import TradingPipeline  # noqa: E402
from tradingai.core.signal import Direction  # noqa: E402
from tradingai.mt5.connector import MT5Connector  # noqa: E402
from tradingai.mt5.data_feed import CandleCloseWatcher  # noqa: E402
from tradingai.mt5.order_executor import OrderExecutor  # noqa: E402
from tradingai.mt5.risk_manager import RiskManager  # noqa: E402
from tradingai.mt5.scaled_exit import should_move_to_breakeven, should_take_partial_profit  # noqa: E402
from tradingai.mt5.structure_exit import compute_dynamic_take_profit, structure_invalidated  # noqa: E402
from tradingai.mt5.trade_log import append_trade_event  # noqa: E402
from tradingai.mt5.trailing_stop import compute_trailing_sl  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402

from loguru import logger  # noqa: E402

RETRY_SECONDS = 15


def _within_trading_hours_now(trading_hours_utc: tuple[int, int] | None) -> bool:
    """Mismo criterio que RiskManager._within_trading_hours, pero contra la hora
    ACTUAL en vez de la de un signal -- para no correr la prediccion completa
    (fetch de 4 temporalidades + ensemble GBM) en ciclos que se van a rechazar de
    todos modos. Antes de esto, los 14 procesos calculaban una prediccion cada 15
    min tambien fuera de horario (ej. toda la sesion asiatica en solitario) solo
    para descartarla en RiskManager -- CPU/calor desperdiciado sin ningun efecto en
    que se abra o no una operacion (esa logica no cambia, sigue en RiskManager).
    """
    if trading_hours_utc is None:
        return True
    start, end = trading_hours_utc
    hour = datetime.now(timezone.utc).hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _maybe_trail_stop(
    connector: MT5Connector,
    symbol: str,
    position: dict,
    initial_risk: float,
    trailing_config: dict,
    trades_log_path: str | None,
) -> None:
    """Mueve el SL a la ultima zona de estructura (swing) si la operacion ya alcanzo
    el multiplo de R configurado en beneficio -- ver tradingai.mt5.trailing_stop.

    `initial_risk` es el riesgo ORIGINAL (entry-SL al abrir), capturado una sola vez
    por el llamador -- NO `position["sl"]` actual, que puede haber cambiado (por este
    mismo trailing en un ciclo anterior, o por un cierre parcial que lo movio a
    breakeven, ver `_maybe_move_to_breakeven`).
    """
    ticket = position["ticket"]
    try:
        direction = Direction.LONG if position["type"] == "buy" else Direction.SHORT
        tick = connector.get_symbol_tick(symbol)
        current_price = tick["bid"] if direction == Direction.LONG else tick["ask"]
        candles = connector.get_candles(symbol, "M15", trailing_config.get("swing_lookback_candles", 100))
        new_sl = compute_trailing_sl(
            candles,
            direction,
            entry_price=position["price_open"],
            initial_risk=initial_risk,
            current_sl=position["sl"],
            current_price=current_price,
            r_multiple_to_activate=trailing_config.get("activate_at_r_multiple", 1.0),
            swing_left=trailing_config.get("swing_left", 3),
            swing_right=trailing_config.get("swing_right", 3),
        )
    except Exception:
        logger.exception(f"[{symbol}] Error calculando trailing stop para ticket {ticket}")
        return

    if new_sl is None:
        return

    try:
        connector.modify_position_sl(ticket, new_sl)
    except Exception:
        logger.exception(f"[{symbol}] Error moviendo SL a {new_sl} para ticket {ticket}")
        return

    logger.info(f"[{symbol}] SL AJUSTADO (trailing a zona clave) ticket={ticket} nuevo_sl={new_sl}")
    if trades_log_path:
        append_trade_event(
            trades_log_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event="AJUSTE_SL",
            ticket=ticket,
            symbol=symbol,
            direction=position["type"],
            sl=new_sl,
        )


def _maybe_take_partial_profit(
    connector: MT5Connector,
    symbol: str,
    position: dict,
    scaled_exit_config: dict,
    trades_log_path: str | None,
) -> None:
    """Cierra una fraccion de la posicion al alcanzar un primer TP conservador (a
    mitad de camino hacia el TP original por defecto) y deja el resto corriendo,
    gestionado por el trailing stop -- ver tradingai.mt5.scaled_exit."""
    ticket = position["ticket"]
    tp1_fraction = scaled_exit_config.get("tp1_fraction", 0.5)
    close_fraction = scaled_exit_config.get("close_fraction", 0.5)

    try:
        direction = Direction.LONG if position["type"] == "buy" else Direction.SHORT
        tick = connector.get_symbol_tick(symbol)
        current_price = tick["bid"] if direction == Direction.LONG else tick["ask"]
        reached_tp1 = should_take_partial_profit(
            direction, position["price_open"], position["tp"], current_price, tp1_fraction
        )
    except Exception:
        logger.exception(f"[{symbol}] Error evaluando TP1 para ticket {ticket}")
        return

    if not reached_tp1:
        return

    try:
        # Idempotencia sin estado propio: si ya hubo un cierre (parcial o total)
        # antes -- incluso en una corrida anterior del proceso, ej. tras un
        # reinicio -- el historial de deals ya tiene una salida (entry==1) aunque
        # la posicion siga abierta con volumen reducido.
        deals = connector.get_position_history(ticket)
        if any(d["entry"] == 1 for d in deals):
            return

        symbol_info = connector.get_symbol_info(symbol)
        volume_step = symbol_info.get("volume_step", 0.01) or 0.01
        volume_min = symbol_info.get("volume_min", volume_step)
        close_volume = round(round((position["volume"] * close_fraction) / volume_step) * volume_step, 8)
        if close_volume < volume_min or close_volume >= position["volume"]:
            return

        connector.close_position(ticket, volume=close_volume)
    except Exception:
        logger.exception(f"[{symbol}] Error tomando ganancia parcial para ticket {ticket}")
        return

    logger.info(f"[{symbol}] CIERRE PARCIAL (TP1 alcanzado) ticket={ticket} volumen={close_volume}")
    if trades_log_path:
        append_trade_event(
            trades_log_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event="CIERRE_PARCIAL",
            ticket=ticket,
            symbol=symbol,
            direction=position["type"],
            lot_size=close_volume,
        )


def _maybe_move_to_breakeven(
    connector: MT5Connector,
    symbol: str,
    position: dict,
    trades_log_path: str | None,
) -> float | None:
    """Si la posicion ya tuvo un cierre parcial (TP1) y el SL todavia no mejoro mas
    alla de breakeven (ni por este mismo mecanismo en un ciclo anterior, ni por el
    trailing), lo fuerza a breakeven -- asegura que, tras bancar la mitad en TP1, el
    resto no pueda terminar en perdida neta. Si el trailing ya lo supero (SL mejor
    que breakeven), no se toca -- pedido explicito del usuario el 2026-08-26, nunca
    aflojar lo que el trailing ya movio.

    Corre en CADA ciclo (no solo el del cierre parcial): si el trailing tarda varios
    ciclos en encontrar una zona de estructura, este mecanismo sigue protegiendo el
    breakeven mientras tanto, y se auto-desactiva en cuanto el trailing lo supera.

    Devuelve el nuevo SL si lo cambio (o None si no hizo falta) -- el llamador debe
    actualizar `position["sl"]` con este valor antes de que el trailing corra en el
    MISMO ciclo (ver bug real del 2026-08-27: sin esto, el trailing leia el SL viejo
    de antes de este ajuste y podia "mejorar" a un valor en realidad peor que el
    breakeven recien puesto).
    """
    ticket = position["ticket"]
    try:
        deals = connector.get_position_history(ticket)
        already_had_a_close = any(d["entry"] == 1 for d in deals)
        if not already_had_a_close:
            return None

        direction = Direction.LONG if position["type"] == "buy" else Direction.SHORT
        entry = position["price_open"]
        if not should_move_to_breakeven(direction, entry, position["sl"]):
            return None
    except Exception:
        logger.exception(f"[{symbol}] Error evaluando breakeven para ticket {ticket}")
        return None

    try:
        connector.modify_position_sl(ticket, entry)
    except Exception:
        logger.exception(f"[{symbol}] Error moviendo SL a breakeven para ticket {ticket}")
        return None

    logger.info(f"[{symbol}] SL A BREAKEVEN (tras cierre parcial) ticket={ticket} nuevo_sl={entry}")
    if trades_log_path:
        append_trade_event(
            trades_log_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event="AJUSTE_SL",
            ticket=ticket,
            symbol=symbol,
            direction=position["type"],
            sl=entry,
        )
    return entry


def _maybe_close_on_structure_invalidation(
    connector: MT5Connector,
    symbol: str,
    position: dict,
    structure_exit_config: dict,
    trades_log_path: str | None,
    cooldown_state: dict | None = None,
    opened_at: datetime | None = None,
) -> bool:
    """Cierra la posicion completa si la secuencia de swings (estructura de precio
    real, ver `core.structure`) se rompio en contra de la direccion de la operacion
    -- no hace falta esperar a que el precio llegue al SL fijo. Redefinido el
    2026-08-28 (antes usaba el apilamiento de EMA20/50/200, pedido explicito del
    usuario de usar estructura de velas real en vez de una media movil). Ver
    tradingai.mt5.structure_exit.

    Validado con backtest purgado antes de encenderse en vivo (ver
    scripts/backtest_structure_exit.py, 2026-08-27): expectancy_r 0.45 -> 1.09 sobre
    3204 operaciones en EURUSD/GBPJPY/GBPAUD/USDCHF, mejora consistente en los 4.

    Devuelve True si cerro la posicion -- el llamador debe saltarse el resto de la
    gestion de ESTE ticket en el mismo ciclo (ya no existe).

    Si se pasa `cooldown_state` (dict compartido con el bucle principal de este
    mismo proceso, ver `main()`), marca la hora del cierre para que no se vuelva a
    abrir en este simbolo por un rato -- caso real del 2026-08-27: AUDUSD reabrio la
    misma apuesta (short) 6 veces en unas horas contra un sesgo alcista estable, y
    cada reapertura pagaba spread de nuevo solo para cerrarse casi al instante otra
    vez. El cierre en si ya es barato (spread, no el SL completo) -- esto evita
    pagarlo repetidas veces mientras el desacuerdo modelo/estructura no cambia.

    `opened_at` acota cuando se permite evaluar esto -- bug real encontrado en vivo
    el 2026-08-28: el watcher sondea cada 30s, asi que el PRIMER chequeo tras abrir
    puede caer sobre la MISMA vela M15 que genero la señal de entrada (la siguiente
    vela todavia no cierra). Eso no es "la estructura cambio", es que el sesgo ya
    estaba en contra desde el primer momento -- el backtest de validacion NUNCA
    probo ese caso (`_simulate_trade_dynamic` siempre arranca en la vela SIGUIENTE a
    la entrada, nunca en la misma). Se exige al menos `min_hold_minutes` desde que
    se vio la posicion por primera vez (una vela M15 nueva) antes de poder invalidar
    por estructura, para igualar lo que realmente se valido.
    """
    ticket = position["ticket"]
    min_hold_minutes = structure_exit_config.get("min_hold_minutes", 15)
    if opened_at is not None and datetime.now(timezone.utc) - opened_at < timedelta(minutes=min_hold_minutes):
        return False

    try:
        direction = Direction.LONG if position["type"] == "buy" else Direction.SHORT
        candles = connector.get_candles(symbol, "M15", structure_exit_config.get("swing_lookback_candles", 100))
        invalidated = structure_invalidated(
            candles, direction,
            swing_left=structure_exit_config.get("swing_left", 3),
            swing_right=structure_exit_config.get("swing_right", 3),
        )
    except Exception:
        logger.exception(f"[{symbol}] Error evaluando invalidacion de estructura para ticket {ticket}")
        return False

    if not invalidated:
        return False

    try:
        connector.close_position(ticket)
    except Exception:
        logger.exception(f"[{symbol}] Error cerrando por invalidacion de estructura ticket {ticket}")
        return False

    if cooldown_state is not None:
        cooldown_minutes = structure_exit_config.get("reentry_cooldown_minutes", 30)
        cooldown_state["until"] = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)

    logger.info(f"[{symbol}] CIERRE POR ESTRUCTURA (sesgo invalidado) ticket={ticket}")
    if trades_log_path:
        append_trade_event(
            trades_log_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event="CIERRE_ESTRUCTURA",
            ticket=ticket,
            symbol=symbol,
            direction=position["type"],
        )
    return True


def _maybe_extend_take_profit(
    connector: MT5Connector,
    symbol: str,
    position: dict,
    structure_exit_config: dict,
    trades_log_path: str | None,
) -> None:
    """Extiende el TP hacia el siguiente swing de estructura si el sesgo se
    mantiene a favor -- deja correr al ganador (1:3, 1:4...) en vez de cerrar en un
    multiplo fijo de riesgo. Ratchet: solo ALEJA el TP, nunca lo acerca (ver
    tradingai.mt5.structure_exit)."""
    ticket = position["ticket"]
    try:
        direction = Direction.LONG if position["type"] == "buy" else Direction.SHORT
        tick = connector.get_symbol_tick(symbol)
        current_price = tick["bid"] if direction == Direction.LONG else tick["ask"]
        candles = connector.get_candles(symbol, "M15", structure_exit_config.get("swing_lookback_candles", 100))
        new_tp = compute_dynamic_take_profit(
            candles,
            direction,
            current_tp=position["tp"],
            current_price=current_price,
            swing_left=structure_exit_config.get("swing_left", 3),
            swing_right=structure_exit_config.get("swing_right", 3),
        )
    except Exception:
        logger.exception(f"[{symbol}] Error calculando TP dinamico para ticket {ticket}")
        return

    if new_tp is None:
        return

    try:
        connector.modify_position_sl(ticket, sl=position["sl"], tp=new_tp)
    except Exception:
        logger.exception(f"[{symbol}] Error extendiendo TP a {new_tp} para ticket {ticket}")
        return

    logger.info(f"[{symbol}] TP EXTENDIDO (estructura a favor) ticket={ticket} nuevo_tp={new_tp}")
    if trades_log_path:
        append_trade_event(
            trades_log_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event="AJUSTE_TP",
            ticket=ticket,
            symbol=symbol,
            direction=position["type"],
            tp=new_tp,
        )


def _position_watcher_loop(
    connector: MT5Connector,
    symbol: str,
    stop_event: threading.Event,
    trades_log_path: str | None = None,
    poll_seconds: int = 30,
    trailing_config: dict | None = None,
    scaled_exit_config: dict | None = None,
    structure_exit_config: dict | None = None,
    structure_cooldown_state: dict | None = None,
    loss_breaker_config: dict | None = None,
    consecutive_loss_state: dict | None = None,
) -> None:
    """Detecta cierres de posiciones (SL/TP ejecutado por el broker) que no pasan por
    `TradingPipeline.run_once()` -- este puede tardar hasta un ciclo de vela en
    enterarse de una entrada nueva, pero un cierre puede pasar en cualquier momento.
    Corre en un hilo aparte con poll corto, independiente del cierre de velas.

    En el mismo poll tambien gestiona el trailing stop (ver `_maybe_trail_stop`) y la
    salida escalonada (ver `_maybe_take_partial_profit`) de las posiciones que siguen
    abiertas, si `trailing_config`/`scaled_exit_config` lo tienen habilitado.
    """
    # Riesgo ORIGINAL (entry-SL) de cada posicion, capturado la primera vez que se ve
    # cada ticket -- NO recalculado de `position["sl"]` en cada ciclo, porque el SL
    # puede cambiar (trailing, o breakeven tras cierre parcial) sin que eso deba
    # alterar el "1R" que activa el trailing (ver tradingai.mt5.trailing_stop).
    entry_risk_by_ticket: dict[int, float] = {}
    # Hora en que este proceso vio cada ticket por primera vez -- proxy de la hora de
    # apertura (el watcher sondea cada `poll_seconds`, asi que el margen de error es
    # ese, insignificante frente al `min_hold_minutes` de mas abajo). Usado para no
    # evaluar invalidacion de estructura sobre la MISMA vela que genero la entrada --
    # ver docstring de _maybe_close_on_structure_invalidation (bug real 2026-08-28).
    opened_at_by_ticket: dict[int, datetime] = {}

    try:
        initial_positions = [p for p in connector.get_open_positions() if p["symbol"] == symbol]
    except Exception:
        logger.exception(f"[{symbol}] Error obteniendo posiciones iniciales para el monitor de cierres")
        initial_positions = []
    known_tickets = {p["ticket"] for p in initial_positions}
    for p in initial_positions:
        if p.get("sl"):
            entry_risk_by_ticket[p["ticket"]] = abs(p["price_open"] - p["sl"])
        opened_at_by_ticket[p["ticket"]] = datetime.now(timezone.utc)

    while not stop_event.wait(poll_seconds):
        try:
            open_positions = [p for p in connector.get_open_positions() if p["symbol"] == symbol]
        except Exception:
            logger.exception(f"[{symbol}] Error consultando posiciones abiertas")
            continue
        current_tickets = {p["ticket"] for p in open_positions}

        for position in open_positions:
            if position["ticket"] not in entry_risk_by_ticket and position.get("sl"):
                entry_risk_by_ticket[position["ticket"]] = abs(position["price_open"] - position["sl"])
            if position["ticket"] not in opened_at_by_ticket:
                opened_at_by_ticket[position["ticket"]] = datetime.now(timezone.utc)

        for ticket in known_tickets - current_tickets:
            entry_risk_by_ticket.pop(ticket, None)
            opened_at_by_ticket.pop(ticket, None)
            profit, close_price = None, None
            try:
                deals = connector.get_position_history(ticket)
                closing_deals = [d for d in deals if d["entry"] == 1]
                profit = sum(d["profit"] for d in closing_deals)
                close_price = closing_deals[-1]["price"] if closing_deals else None
            except Exception:
                logger.exception(f"[{symbol}] Error obteniendo historial de la posicion {ticket}")
            logger.info(f"[{symbol}] CERRADA ticket={ticket} close_price={close_price} profit={profit}")
            if trades_log_path:
                append_trade_event(
                    trades_log_path,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event="CIERRE",
                    ticket=ticket,
                    symbol=symbol,
                    price=close_price,
                    profit=profit,
                )

            if loss_breaker_config and loss_breaker_config.get("enabled") and consecutive_loss_state is not None and profit is not None:
                if profit < 0:
                    consecutive_loss_state["count"] += 1
                    max_losses = loss_breaker_config.get("max_consecutive_losses", 3)
                    if consecutive_loss_state["count"] >= max_losses:
                        cooldown_minutes = loss_breaker_config.get("cooldown_minutes", 120)
                        consecutive_loss_state["until"] = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)
                        logger.warning(
                            f"[{symbol}] CORTACIRCUITO: {consecutive_loss_state['count']} perdidas consecutivas seguidas, "
                            f"pausando entradas nuevas hasta {consecutive_loss_state['until']}"
                        )
                elif profit > 0:
                    consecutive_loss_state["count"] = 0

        # Cierre anticipado por invalidacion de estructura: PAUSADO el 2026-08-28 con
        # datos en vivo (14 cierres, -$332 realizado; de los 10 que no habrian tocado
        # su SL, dejarlos correr mostraria +$2600 combinados ahora mismo) -- el
        # backtest del 27 mostro mejora agregada, pero la evidencia en vivo de hoy la
        # contradice. Se apaga con `invalidation_close_enabled: false` sin tocar la
        # extension de TP (sin este riesgo, solo aleja el TP, nunca cierra antes de
        # tiempo) mientras se hace el analisis mas riguroso del fin de semana sobre
        # los 10 simbolos que todavia no se validaron.
        if (
            structure_exit_config
            and structure_exit_config.get("enabled")
            and structure_exit_config.get("invalidation_close_enabled", True)
        ):
            closed_by_structure = set()
            for position in open_positions:
                if _maybe_close_on_structure_invalidation(
                    connector, symbol, position, structure_exit_config, trades_log_path, structure_cooldown_state,
                    opened_at_by_ticket.get(position["ticket"]),
                ):
                    closed_by_structure.add(position["ticket"])
            if closed_by_structure:
                # Ya no existen -- se saltan del resto de la gestion de este mismo
                # ciclo (parcial/breakeven/trailing/extension de TP no tienen sentido
                # sobre un ticket recien cerrado). El diff de cierres del PROXIMO
                # ciclo los registrara igual como CIERRE con su profit real.
                open_positions = [p for p in open_positions if p["ticket"] not in closed_by_structure]

        if scaled_exit_config and scaled_exit_config.get("enabled"):
            for position in open_positions:
                _maybe_take_partial_profit(connector, symbol, position, scaled_exit_config, trades_log_path)
            if scaled_exit_config.get("move_sl_to_breakeven"):
                for position in open_positions:
                    new_sl = _maybe_move_to_breakeven(connector, symbol, position, trades_log_path)
                    if new_sl is not None:
                        # Mantener el dict en memoria sincronizado para que el
                        # trailing (mismo ciclo, corre justo despues) compare contra
                        # el SL real actual, no contra el valor de antes de este
                        # ajuste -- ver docstring de _maybe_move_to_breakeven.
                        position["sl"] = new_sl

        if trailing_config and trailing_config.get("enabled"):
            for position in open_positions:
                initial_risk = entry_risk_by_ticket.get(position["ticket"])
                if initial_risk is None:
                    continue
                _maybe_trail_stop(connector, symbol, position, initial_risk, trailing_config, trades_log_path)

        if structure_exit_config and structure_exit_config.get("enabled"):
            for position in open_positions:
                _maybe_extend_take_profit(connector, symbol, position, structure_exit_config, trades_log_path)

        known_tickets = current_tickets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()

    config = get_settings()
    secrets = config["secrets"]
    setup_logging(secrets.log_level, config["paths"]["logs_dir"])

    if secrets.trading_mode == "live":
        logger.warning("TRADING_MODE=live: se ejecutaran ordenes REALES en la cuenta configurada.")

    connector = MT5Connector(base_url=secrets.mt5_bridge_url)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.suffix == ".joblib":
        predictor = GBMPredictor.from_checkpoint(checkpoint_path)
    else:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        feature_columns = checkpoint.get("feature_columns")
        if feature_columns is None:
            raise RuntimeError("El checkpoint no incluye feature_columns; reentrena con la version actual de train.py.")
        predictor = Predictor.from_checkpoint(checkpoint_path, feature_columns)
    trading_hours = config["trading"].get("trading_hours_utc")
    risk_manager = RiskManager(
        risk_per_trade_pct=secrets.risk_per_trade_pct,
        # config.yaml es la fuente de verdad para el resto de parametros de riesgo de
        # esta llamada -- max_open_positions vivia en Secrets/.env (nunca configurado
        # ahi en la practica, .env ni siquiera existe) mientras config.yaml tenia su
        # propio valor que jamas se leia. Bug real encontrado el 2026-08-27: cambiar
        # config.yaml no tenia ningun efecto en este parametro. `secrets.max_open_positions`
        # queda solo como fallback por si config.yaml no lo define.
        max_open_positions=config["trading"].get("max_open_positions", secrets.max_open_positions),
        max_open_positions_high_confidence=config["trading"].get("max_open_positions_high_confidence"),
        high_confidence_override=config["trading"].get("high_confidence_override", 0.90),
        min_risk_reward_ratio=config["trading"].get("min_risk_reward_ratio", 2.0),
        max_daily_drawdown_pct=config["trading"].get("max_daily_drawdown_pct"),
        trading_hours_utc=tuple(trading_hours) if trading_hours else None,
        max_correlated_same_direction=config["trading"].get("max_correlated_same_direction", 2),
        max_positions_per_symbol=config["trading"].get("max_positions_per_symbol", 1),
        max_portfolio_risk_pct=config["trading"].get("max_portfolio_risk_pct"),
        news_calendar_config=config["trading"].get("news_calendar"),
        min_sl_spread_multiple=config["trading"].get("min_sl_spread_multiple"),
        connector=connector,
    )

    trades_log_path = Path(config["paths"]["logs_dir"]) / "live" / "trades.csv"

    with connector:
        executor = OrderExecutor(
            connector,
            risk_manager,
            max_margin_pct_per_trade=config["trading"].get("max_margin_pct_per_trade", 10.0),
            trades_log_path=trades_log_path,
        )
        pipeline = TradingPipeline(
            connector, predictor, risk_manager, executor,
            confidence_threshold=config["model"]["outputs"].get("confidence_threshold", 0.6),
        )
        candle_state_file = Path(config["paths"]["logs_dir"]) / "live" / ".candle_state" / f"{args.symbol}.txt"
        watcher = CandleCloseWatcher(connector, args.symbol, ANCHOR_TIMEFRAME, state_file=candle_state_file)

        # Compartido con el watcher (mismo proceso, un solo hilo por simbolo): marca
        # hasta cuando no reabrir tras un cierre por invalidacion de estructura -- ver
        # docstring de _maybe_close_on_structure_invalidation (caso real AUDUSD
        # 2026-08-27, reabria la misma apuesta perdedora cada 15-20 min).
        structure_cooldown_state: dict = {"until": None}

        # Cortacircuito por perdidas consecutivas (2026-08-28, caso real EURJPY:
        # 7 aperturas seguidas en el mismo dia, todas en la misma direccion, -$317
        # neto) -- a diferencia del cooldown de arriba (especifico del cierre por
        # estructura), este cuenta CUALQUIER cierre en perdida seguido, sin importar
        # el motivo (SL, estructura, lo que sea). Se resetea con una ganancia.
        consecutive_loss_state: dict = {"count": 0, "until": None}

        stop_event = threading.Event()
        watcher_thread = threading.Thread(
            target=_position_watcher_loop,
            args=(connector, args.symbol, stop_event, trades_log_path),
            kwargs={
                "trailing_config": config["trading"].get("trailing_stop"),
                "scaled_exit_config": config["trading"].get("scaled_exit"),
                "structure_exit_config": config["trading"].get("structure_exit"),
                "structure_cooldown_state": structure_cooldown_state,
                "loss_breaker_config": config["trading"].get("consecutive_loss_breaker"),
                "consecutive_loss_state": consecutive_loss_state,
            },
            daemon=True,
        )
        watcher_thread.start()

        logger.info(f"Iniciando bucle en vivo: {args.symbol} (mode={secrets.trading_mode})")
        try:
            while True:
                try:
                    watcher.wait_for_new_candle()
                    cooldown_until = structure_cooldown_state.get("until")
                    if cooldown_until is not None and datetime.now(timezone.utc) < cooldown_until:
                        logger.debug(f"[{args.symbol}] En cooldown tras cierre por estructura hasta {cooldown_until}, se salta este ciclo")
                        continue
                    loss_breaker_until = consecutive_loss_state.get("until")
                    if loss_breaker_until is not None and datetime.now(timezone.utc) < loss_breaker_until:
                        logger.debug(f"[{args.symbol}] Cortacircuito activo por perdidas consecutivas hasta {loss_breaker_until}, se salta este ciclo")
                        continue
                    if not _within_trading_hours_now(tuple(trading_hours) if trading_hours else None):
                        logger.debug(f"[{args.symbol}] Fuera del horario permitido {trading_hours} UTC, no se calcula prediccion este ciclo")
                        continue
                    pipeline.run_once(args.symbol)
                except requests.exceptions.RequestException:
                    # El bridge puede caerse un momento (reinicio para cargar codigo
                    # nuevo, hipo de Wine) -- sin este catch, un solo fallo de red
                    # tumbaba el proceso entero en vez de reintentar en el siguiente
                    # ciclo (visto en vivo el 2026-08-24: los 5 procesos del piloto
                    # murieron al reiniciar el bridge para anadir simbolos nuevos).
                    logger.warning(f"[{args.symbol}] Bridge no disponible, reintentando en {RETRY_SECONDS}s")
                    time.sleep(RETRY_SECONDS)
        finally:
            stop_event.set()


if __name__ == "__main__":
    main()
