"""Walk-forward validation: entrena con ventana expansiva y evalua en el siguiente
tramo nunca visto, repetido para varios folds.

Sirve para saber si el edge medido en un unico split 80/20 (~1 mes de datos) es señal
real o suerte de ese periodo especifico. Cada fold entrena un modelo desde cero sobre
todos los datos anteriores a ese fold (ventana expansiva), reservando el ultimo tramo
del propio training para early stopping interno, y evalua sobre el fold siguiente
(nunca visto por ese modelo).

Uso:
    python scripts/walk_forward.py --symbol EURUSD --n-folds 5
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
import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, last_closed_bar_index  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.evaluation.backtester import Backtester, summarize  # noqa: E402
from tradingai.ai.inference.predictor import Predictor  # noqa: E402
from tradingai.ai.models.base import build_model  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.ai.training.trainer import Trainer  # noqa: E402
from tradingai.core.pipeline import FEATURE_WARMUP_BARS  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402

from loguru import logger  # noqa: E402


def _pip_size(symbol: str) -> float:
    if symbol.upper().endswith("JPY") or symbol.upper() in {"XAUUSD", "XAGUSD"}:
        return 0.01
    return 0.0001


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--n-folds", type=int, default=5, help="Folds totales; se evaluan n_folds-1 (el 1o solo entrena)")
    parser.add_argument("--internal-val-split", type=float, default=0.15, help="Fraccion del training expandido reservada para early stopping")
    parser.add_argument("--step", type=int, default=20, help="cada cuantos anchors M15 se genera una senal al evaluar")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Umbral para medir calidad (0.0 = todas las señales direccionales)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para pesos iniciales y barajado")
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    data_dir = Path(args.data_dir or config["paths"]["raw_data"])
    candles_by_tf = {tf: load_csv(data_dir / f"{args.symbol}_{tf}.csv") for tf in TIMEFRAMES}
    features_by_tf = {tf: normalize_ohlcv(build_feature_pipeline(candles_by_tf[tf], config)) for tf in TIMEFRAMES}
    feature_columns = select_feature_columns(features_by_tf["M15"])

    seq_len_by_tf = config["model"]["sequence_length"]
    dataset = MultiTimeframeTradingDataset(
        features_by_tf, feature_columns, seq_len_by_tf,
        tp_atr_mult=config["model"]["tp_atr_mult"], sl_atr_mult=config["model"]["sl_atr_mult"],
    )
    n = len(dataset)
    logger.info(f"[{args.symbol}] Dataset total: {n} ejemplos alineados")

    fold_bounds = [int(n * i / args.n_folds) for i in range(args.n_folds + 1)]
    logger.info(f"Limites de fold (indices en el dataset): {fold_bounds}")

    m15_raw = candles_by_tf["M15"]
    m15_raw_ts = m15_raw["timestamp"].to_numpy()
    cutoff_idx = {tf: last_closed_bar_index(candles_by_tf[tf]["timestamp"].to_numpy(), m15_raw_ts) for tf in TIMEFRAMES}
    max_window = {tf: sl + FEATURE_WARMUP_BARS for tf, sl in seq_len_by_tf.items()}
    backtest_cfg = config.get("backtest", {})

    results = []
    for fold in range(1, args.n_folds):
        test_start_ds_idx = fold_bounds[fold]
        test_end_ds_idx = fold_bounds[fold + 1]
        # Purge (Lopez de Prado): la etiqueta de un ejemplo mira `dataset.horizon`
        # velas hacia adelante -- el val_interno, al ser contiguo con test_start, es
        # el que puede filtrar informacion de test si no se recorta antes. Nunca se
        # entrena/valida sobre los ultimos `horizon` ejemplos antes de test_start.
        train_end = test_start_ds_idx - dataset.horizon
        internal_val_start = int(train_end * (1 - args.internal_val_split))

        train_subset = Subset(dataset, range(0, internal_val_start))
        internal_val_subset = Subset(dataset, range(internal_val_start, train_end))

        class_counts = np.bincount(dataset.direction[:internal_val_start], minlength=3)
        class_weights = class_counts.sum() / (3 * np.maximum(class_counts, 1))

        logger.info(
            f"=== Fold {fold}/{args.n_folds - 1} ===  "
            f"train=[0:{internal_val_start}] val_interno=[{internal_val_start}:{train_end}] "
            f"test=[{test_start_ds_idx}:{test_end_ds_idx}] ({test_end_ds_idx - test_start_ds_idx} ejemplos)"
        )

        model = build_model(config, n_features=len(feature_columns))
        trainer = Trainer(model, config, direction_class_weights=torch.tensor(class_weights, dtype=torch.float32))

        t0 = time.time()
        trainer.fit(train_subset, internal_val_subset)
        logger.info(f"Fold {fold} entrenado en {(time.time() - t0) / 60:.1f} min")

        predictor = Predictor(model, config, feature_columns)

        test_start_m15 = int(dataset.anchor_positions[test_start_ds_idx])
        test_end_m15 = int(dataset.anchor_positions[test_end_ds_idx - 1]) + 1

        signals = []
        for i in range(test_start_m15, test_end_m15, args.step):
            if any(cutoff_idx[tf][i] < 0 for tf in TIMEFRAMES):
                continue
            window_by_tf = {
                tf: candles_by_tf[tf].iloc[max(0, cutoff_idx[tf][i] + 1 - max_window[tf]) : cutoff_idx[tf][i] + 1]
                for tf in TIMEFRAMES
            }
            try:
                signals.append((i, predictor.predict(window_by_tf, symbol=args.symbol)))
            except ValueError:
                continue

        backtester = Backtester(
            confidence_threshold=args.confidence_threshold,
            max_holding_bars=backtest_cfg.get("max_holding_bars", 50),
            spread_pips=backtest_cfg.get("spread_pips", 1.0),
            slippage_pips=backtest_cfg.get("slippage_pips", 0.2),
            commission_pips=backtest_cfg.get("commission_pips", 0.0),
            pip_size=_pip_size(args.symbol),
        )
        trades = backtester.run(m15_raw, signals)
        stats = summarize(trades)
        stats["fold"] = fold
        stats["test_start"] = str(m15_raw["timestamp"].iloc[test_start_m15])
        stats["test_end"] = str(m15_raw["timestamp"].iloc[min(test_end_m15, len(m15_raw) - 1)])
        results.append(stats)

        if stats["n_trades"] > 0:
            logger.info(
                f"Fold {fold} ({stats['test_start']} -> {stats['test_end']}): "
                f"{stats['n_trades']} ops, win_rate={stats['win_rate'] * 100:.1f}%, "
                f"pnl={stats['total_pnl_pct'] * 100:.2f}%"
            )
        else:
            logger.info(f"Fold {fold}: 0 operaciones")

    logger.info("=== Resumen walk-forward ===")
    for r in results:
        if r["n_trades"] > 0:
            logger.info(
                f"Fold {r['fold']} ({r['test_start']} -> {r['test_end']}): "
                f"{r['n_trades']} ops, win_rate={r['win_rate'] * 100:.1f}%, pnl={r['total_pnl_pct'] * 100:.2f}%"
            )
        else:
            logger.info(f"Fold {r['fold']}: 0 operaciones")

    win_rates = [r["win_rate"] for r in results if r["n_trades"] > 0]
    pnls = [r["total_pnl_pct"] for r in results if r["n_trades"] > 0]
    if win_rates:
        logger.info(
            f"Win rate: media {np.mean(win_rates) * 100:.1f}% (std {np.std(win_rates) * 100:.1f}%) "
            f"entre {len(win_rates)} folds"
        )
        logger.info(f"PnL total acumulado (todos los folds): {sum(pnls) * 100:.2f}%")


if __name__ == "__main__":
    main()
