"""Compara el backtest ESTATICO (TP/SL fijo 2:1) contra el modo `dynamic_exit` de
`ai.evaluation.backtester.Backtester` (cierre anticipado por invalidacion de
estructura + extension de TP hacia el siguiente swing) -- ver conversacion
2026-08-27: antes de encender esto en el piloto en vivo, se valida con historico
usando las MISMAS señales del modelo en ambos brazos (solo cambia la logica de
salida), igual criterio que se exige para cualquier cambio del modelo de entrada.

Uso:
    python scripts/backtest_structure_exit.py --symbols EURUSD GBPJPY GBPAUD USDCHF
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import threadpoolctl  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, flatten_last_timestep  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.evaluation.backtester import Backtester, pip_size, resolve_spread_pips, summarize  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.core.signal import Direction, TradingSignal  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402
from tradingai.utils.thermal import wait_for_safe_temp  # noqa: E402

from loguru import logger  # noqa: E402

_DIRECTION_MAP = {0: Direction.NEUTRAL, 1: Direction.LONG, 2: Direction.SHORT}


def _run_symbol(symbol: str, args: argparse.Namespace, config: dict) -> tuple[dict, dict]:
    data_dir = Path(args.data_dir or config["paths"]["raw_data"])
    candles_by_tf = {tf: load_csv(data_dir / f"{symbol}_{tf}.csv") for tf in TIMEFRAMES}
    features_by_tf = {tf: normalize_ohlcv(build_feature_pipeline(candles_by_tf[tf], config)) for tf in TIMEFRAMES}
    feature_columns = select_feature_columns(features_by_tf["M15"])

    tp_atr_mult = config["model"]["tp_atr_mult"]
    sl_atr_mult = config["model"]["sl_atr_mult"]
    seq_len_by_tf = config["model"]["sequence_length"]
    dataset = MultiTimeframeTradingDataset(
        features_by_tf, feature_columns, seq_len_by_tf, tp_atr_mult=tp_atr_mult, sl_atr_mult=sl_atr_mult,
    )
    n = len(dataset)
    X = flatten_last_timestep(dataset.sequences)
    y = dataset.direction

    fold_bounds = [int(n * i / args.n_folds) for i in range(args.n_folds + 1)]

    all_static_trades = []
    all_dynamic_trades = []

    for fold in range(1, args.n_folds):
        train_end = fold_bounds[fold]
        test_start, test_end = fold_bounds[fold], fold_bounds[fold + 1]
        purged_train_end = test_start - dataset.horizon
        X_train, y_train = X[:purged_train_end], y[:purged_train_end]
        X_test = X[test_start:test_end]

        t0 = time.time()
        models = []
        with threadpoolctl.threadpool_limits(limits=args.max_threads):
            for i in range(args.n_seeds):
                wait_for_safe_temp(max_temp_c=args.max_temp_c)
                m = HistGradientBoostingClassifier(random_state=args.seed + i, max_iter=200, early_stopping=True)
                m.fit(X_train, y_train)
                models.append(m)
        train_time = time.time() - t0

        y_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
        y_pred = y_proba.argmax(axis=1)

        m15_features = features_by_tf["M15"]
        anchor_idx_test = dataset.anchor_positions[test_start:test_end]
        strided_local_indices = range(0, len(anchor_idx_test), args.backtest_step)
        signals = []
        for local_i in strided_local_indices:
            m15_row_idx = anchor_idx_test[local_i]
            direction_idx = int(y_pred[local_i])
            direction = _DIRECTION_MAP[direction_idx]
            confidence = float(y_proba[local_i, direction_idx])

            entry_price = float(m15_features["close"].iloc[m15_row_idx])
            atr = float(m15_features["atr_14"].iloc[m15_row_idx])

            take_profit = stop_loss = None
            if direction != Direction.NEUTRAL and atr > 0:
                sign = 1 if direction == Direction.LONG else -1
                take_profit = entry_price + sign * tp_atr_mult * atr
                stop_loss = entry_price - sign * sl_atr_mult * atr

            signals.append((
                m15_row_idx,
                TradingSignal(
                    symbol=symbol, timeframe="M15", timestamp=datetime.now(timezone.utc),
                    direction=direction, confidence=confidence, entry_price=entry_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                ),
            ))

        backtest_cfg = config.get("backtest", {})
        common_kwargs = dict(
            confidence_threshold=args.confidence_threshold,
            max_holding_bars=backtest_cfg.get("max_holding_bars", 50),
            spread_pips=resolve_spread_pips(backtest_cfg, symbol),
            slippage_pips=backtest_cfg.get("slippage_pips", 0.2),
            commission_pips=backtest_cfg.get("commission_pips", 0.0),
            pip_size=pip_size(symbol),
        )
        static_bt = Backtester(**common_kwargs)
        dynamic_bt = Backtester(**common_kwargs, dynamic_exit=True)

        static_trades = static_bt.run(candles_by_tf["M15"], signals)
        dynamic_trades = dynamic_bt.run(candles_by_tf["M15"], signals)
        all_static_trades.extend(static_trades)
        all_dynamic_trades.extend(dynamic_trades)

        logger.info(f"[{symbol}] fold {fold}/{args.n_folds - 1} entrenado en {train_time:.1f}s, "
                    f"{len(static_trades)} operaciones simuladas")

        if args.fold_cooldown_seconds > 0 and fold < args.n_folds - 1:
            time.sleep(args.fold_cooldown_seconds)

    return summarize(all_static_trades), summarize(all_dynamic_trades)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--backtest-step", type=int, default=10)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument("--max-temp-c", type=float, default=80.0)
    parser.add_argument("--fold-cooldown-seconds", type=float, default=15.0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    totals_static = []
    totals_dynamic = []
    for symbol in args.symbols:
        logger.info(f"=== {symbol} ===")
        static_stats, dynamic_stats = _run_symbol(symbol, args, config)
        totals_static.append(static_stats)
        totals_dynamic.append(dynamic_stats)

        logger.info(
            f"[{symbol}] ESTATICO: n={static_stats['n_trades']} win_rate={static_stats.get('win_rate', 0) * 100:.1f}% "
            f"expectancy_r={static_stats.get('expectancy_r', 0):.3f} profit_factor={static_stats.get('profit_factor', 0):.3f}"
        )
        logger.info(
            f"[{symbol}] DINAMICO:  n={dynamic_stats['n_trades']} win_rate={dynamic_stats.get('win_rate', 0) * 100:.1f}% "
            f"expectancy_r={dynamic_stats.get('expectancy_r', 0):.3f} profit_factor={dynamic_stats.get('profit_factor', 0):.3f}"
        )

    n_static = sum(s["n_trades"] for s in totals_static)
    n_dynamic = sum(s["n_trades"] for s in totals_dynamic)
    er_static = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_static) / n_static if n_static else 0.0
    er_dynamic = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_dynamic) / n_dynamic if n_dynamic else 0.0

    logger.info("=== TOTAL AGREGADO (todos los simbolos/folds) ===")
    logger.info(f"ESTATICO: n={n_static} expectancy_r_promedio={er_static:.3f}")
    logger.info(f"DINAMICO:  n={n_dynamic} expectancy_r_promedio={er_dynamic:.3f}")


if __name__ == "__main__":
    main()
