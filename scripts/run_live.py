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
from datetime import datetime, timezone
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
from tradingai.mt5.trade_log import append_trade_event  # noqa: E402
from tradingai.mt5.trailing_stop import compute_trailing_sl  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402

from loguru import logger  # noqa: E402

RETRY_SECONDS = 15


def _maybe_trail_stop(
    connector: MT5Connector,
    symbol: str,
    position: dict,
    trailing_config: dict,
    trades_log_path: str | None,
) -> None:
    """Mueve el SL a la ultima zona de estructura (swing) si la operacion ya alcanzo
    el multiplo de R configurado en beneficio -- ver tradingai.mt5.trailing_stop."""
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


def _position_watcher_loop(
    connector: MT5Connector,
    symbol: str,
    stop_event: threading.Event,
    trades_log_path: str | None = None,
    poll_seconds: int = 30,
    trailing_config: dict | None = None,
) -> None:
    """Detecta cierres de posiciones (SL/TP ejecutado por el broker) que no pasan por
    `TradingPipeline.run_once()` -- este puede tardar hasta un ciclo de vela en
    enterarse de una entrada nueva, pero un cierre puede pasar en cualquier momento.
    Corre en un hilo aparte con poll corto, independiente del cierre de velas.

    En el mismo poll tambien gestiona el trailing stop (ver `_maybe_trail_stop`) de
    las posiciones que siguen abiertas, si `trailing_config` lo tiene habilitado.
    """
    try:
        known_tickets = {p["ticket"] for p in connector.get_open_positions() if p["symbol"] == symbol}
    except Exception:
        logger.exception(f"[{symbol}] Error obteniendo posiciones iniciales para el monitor de cierres")
        known_tickets = set()

    while not stop_event.wait(poll_seconds):
        try:
            open_positions = [p for p in connector.get_open_positions() if p["symbol"] == symbol]
        except Exception:
            logger.exception(f"[{symbol}] Error consultando posiciones abiertas")
            continue
        current_tickets = {p["ticket"] for p in open_positions}

        for ticket in known_tickets - current_tickets:
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

        if trailing_config and trailing_config.get("enabled"):
            for position in open_positions:
                _maybe_trail_stop(connector, symbol, position, trailing_config, trades_log_path)

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
        max_open_positions=secrets.max_open_positions,
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
        watcher = CandleCloseWatcher(connector, args.symbol, ANCHOR_TIMEFRAME)

        stop_event = threading.Event()
        watcher_thread = threading.Thread(
            target=_position_watcher_loop,
            args=(connector, args.symbol, stop_event, trades_log_path),
            kwargs={"trailing_config": config["trading"].get("trailing_stop")},
            daemon=True,
        )
        watcher_thread.start()

        logger.info(f"Iniciando bucle en vivo: {args.symbol} (mode={secrets.trading_mode})")
        try:
            while True:
                try:
                    watcher.wait_for_new_candle()
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
