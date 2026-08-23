"""Bucle en vivo: conecta al bridge MT5, espera cierre de vela, predice y (si aplica) ejecuta.

Uso:
    python scripts/run_live.py --checkpoint data/models/EURUSD_transformer.pt --symbol EURUSD

Requiere el bridge corriendo (scripts/start_mt5_bridge.sh dentro de Wine).
Ver README > "MT5 en Linux".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.multi_timeframe import ANCHOR_TIMEFRAME  # noqa: E402
from tradingai.ai.inference.gbm_predictor import GBMPredictor  # noqa: E402
from tradingai.ai.inference.predictor import Predictor  # noqa: E402
from tradingai.core.pipeline import TradingPipeline  # noqa: E402
from tradingai.mt5.connector import MT5Connector  # noqa: E402
from tradingai.mt5.data_feed import CandleCloseWatcher  # noqa: E402
from tradingai.mt5.order_executor import OrderExecutor  # noqa: E402
from tradingai.mt5.risk_manager import RiskManager  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402

from loguru import logger  # noqa: E402


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
        correlated_groups=config["trading"].get("correlated_groups"),
        max_correlated_same_direction=config["trading"].get("max_correlated_same_direction", 2),
        connector=connector,
    )

    with connector:
        executor = OrderExecutor(connector, risk_manager)
        pipeline = TradingPipeline(connector, predictor, risk_manager, executor)
        watcher = CandleCloseWatcher(connector, args.symbol, ANCHOR_TIMEFRAME)

        logger.info(f"Iniciando bucle en vivo: {args.symbol} (mode={secrets.trading_mode})")
        while True:
            watcher.wait_for_new_candle()
            pipeline.run_once(args.symbol)


if __name__ == "__main__":
    main()
