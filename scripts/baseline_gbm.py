"""Linea base simple (gradient boosting) para saber si hay señal predictiva real en
las features, independientemente de la arquitectura de red neuronal.

En vez de secuencias multi-temporalidad, usa solo el ultimo valor de cada feature por
temporalidad (snapshot actual de D1+H1+M15+M5 concatenado) y un HistGradientBoosting
sobre eso. Entrena en segundos por fold, así que corre los mismos folds de walk-forward
(mismos limites, misma ventana expansiva) para comparar directo contra el transformer.

No hace backtest de trading (no predice entry/tp/sl) — mide directamente si el modelo
acierta la DIRECCION mejor que el azar/la clase mayoritaria, que es la pregunta de fondo:
¿hay señal aqui o no?

Uso:
    python scripts/baseline_gbm.py --symbol EURUSD --n-folds 5
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
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, flatten_last_timestep  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.evaluation.backtester import Backtester, summarize  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.core.signal import Direction, TradingSignal  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402
from tradingai.utils.thermal import wait_for_safe_temp  # noqa: E402

from loguru import logger  # noqa: E402

DIRECTION_NAMES = {0: "neutral", 1: "long", 2: "short"}
_DIRECTION_MAP = {0: Direction.NEUTRAL, 1: Direction.LONG, 2: Direction.SHORT}

def _pip_size(symbol: str) -> float:
    if symbol.upper().endswith("JPY") or symbol.upper() in {"XAUUSD", "XAGUSD"}:
        return 0.01
    return 0.0001


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--internal-val-split", type=float, default=0.15)
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Umbral de prob. maxima para medir precision en subset 'seguro'")
    parser.add_argument("--backtest-step", type=int, default=20, help="Espaciado entre anchors para el backtest (evita operaciones solapadas); la accuracy de clasificacion usa todos los ejemplos igual")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=5, help="Modelos del ensemble por fold (igual que train_gbm.py, para validar lo mismo que corre en produccion)")
    parser.add_argument("--max-threads", type=int, default=1, help="Hilos maximos para sklearn/BLAS -- ver tradingai.utils.thermal, esta maquina se calienta rapido con mas de 1")
    parser.add_argument("--max-temp-c", type=float, default=80.0, help="Pausa antes de entrenar si la CPU supera esta temperatura")
    parser.add_argument("--fold-cooldown-seconds", type=float, default=20.0, help="Pausa fija entre folds para dar tiempo a disipar calor")
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    data_dir = Path(args.data_dir or config["paths"]["raw_data"])
    candles_by_tf = {tf: load_csv(data_dir / f"{args.symbol}_{tf}.csv") for tf in TIMEFRAMES}
    features_by_tf = {tf: normalize_ohlcv(build_feature_pipeline(candles_by_tf[tf], config)) for tf in TIMEFRAMES}
    feature_columns = select_feature_columns(features_by_tf["M15"])

    tp_atr_mult = config["model"]["tp_atr_mult"]
    sl_atr_mult = config["model"]["sl_atr_mult"]

    seq_len_by_tf = config["model"]["sequence_length"]
    dataset = MultiTimeframeTradingDataset(
        features_by_tf, feature_columns, seq_len_by_tf,
        tp_atr_mult=tp_atr_mult, sl_atr_mult=sl_atr_mult,
    )
    n = len(dataset)
    logger.info(f"[{args.symbol}] Dataset total: {n} ejemplos alineados")

    X = flatten_last_timestep(dataset.sequences)
    y = dataset.direction
    logger.info(f"X shape: {X.shape}")

    fold_bounds = [int(n * i / args.n_folds) for i in range(args.n_folds + 1)]
    logger.info(f"Limites de fold: {fold_bounds}")

    fold_accuracies = []
    fold_majority_accuracies = []

    for fold in range(1, args.n_folds):
        train_end = fold_bounds[fold]
        test_start, test_end = fold_bounds[fold], fold_bounds[fold + 1]
        internal_val_start = int(train_end * (1 - args.internal_val_split))

        # Purge (Lopez de Prado, "Advances in Financial Machine Learning"): la
        # etiqueta de un ejemplo mira `dataset.horizon` velas hacia adelante -- un
        # ejemplo justo antes de test_start cuya ventana de etiquetado ya toca datos
        # de test_start en adelante seria una fuga de informacion. Se purga
        # explicitamente (nunca se entrena sobre los ultimos `horizon` ejemplos antes
        # de test_start), sin depender de que --internal-val-split deje un margen
        # suficiente por casualidad (con split=0 no habria margen ninguno).
        purged_train_end = min(internal_val_start, test_start - dataset.horizon)

        X_train, y_train = X[:purged_train_end], y[:purged_train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        # Ensemble de N semillas por fold, igual que scripts/train_gbm.py -- asi el
        # walk-forward valida lo mismo que efectivamente corre en produccion, no un
        # modelo unico mas optimista/pesimista de lo que sera el ensemble real.
        # threadpoolctl fuerza `--max-threads` hilos de forma fiable (las variables
        # de entorno OMP_NUM_THREADS/etc. no siempre alcanzan a limitar sklearn/BLAS
        # una vez el proceso ya arranco -- visto en vivo el 2026-08-25: la CPU llego
        # a 100C en <15s con esas variables puestas). `wait_for_safe_temp` pausa
        # antes de CADA modelo, no solo entre folds, para no dejar que el calor se
        # acumule durante el ensemble entero.
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
        accuracy = (y_pred == y_test).mean()

        majority_class = np.bincount(y_train, minlength=3).argmax()
        majority_accuracy = (y_test == majority_class).mean()

        confident_mask = y_proba.max(axis=1) >= args.confidence_threshold
        confident_accuracy = (y_pred[confident_mask] == y_test[confident_mask]).mean() if confident_mask.any() else float("nan")

        fold_accuracies.append(accuracy)
        fold_majority_accuracies.append(majority_accuracy)

        # Backtest real: mismos multiplicadores ATR fijos que usa el etiquetado, para
        # comparar win_rate/pnl directo contra los folds del transformer (walk_forward.py).
        # Se espacian los anchors (--backtest-step) para no abrir una operacion en cada
        # vela M15 consecutiva -- sin esto, miles de "operaciones" solapadas en el tiempo
        # se suman como si fueran independientes y el pnl total pierde todo sentido.
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

            signals.append(
                (
                    m15_row_idx,
                    TradingSignal(
                        symbol=args.symbol,
                        timeframe="M15",
                        timestamp=datetime.now(timezone.utc),
                        direction=direction,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    ),
                )
            )

        backtest_cfg = config.get("backtest", {})
        backtester = Backtester(
            confidence_threshold=args.confidence_threshold,
            max_holding_bars=backtest_cfg.get("max_holding_bars", 50),
            spread_pips=backtest_cfg.get("spread_pips", 1.0),
            slippage_pips=backtest_cfg.get("slippage_pips", 0.2),
            commission_pips=backtest_cfg.get("commission_pips", 0.0),
            pip_size=_pip_size(args.symbol),
        )
        trades = backtester.run(candles_by_tf["M15"], signals)
        trade_stats = summarize(trades)

        logger.info(
            f"=== Fold {fold}/{args.n_folds - 1} === train={purged_train_end} test={test_end - test_start} "
            f"({train_time:.1f}s)"
        )
        logger.info(
            f"Fold {fold}: accuracy={accuracy * 100:.1f}% (baseline clase mayoritaria={majority_accuracy * 100:.1f}%) "
            f"| confianza>={args.confidence_threshold}: {confident_mask.sum()}/{len(y_test)} casos, "
            f"accuracy en esos={confident_accuracy * 100:.1f}%"
        )
        report = classification_report(
            y_test, y_pred, labels=[0, 1, 2], target_names=["neutral", "long", "short"], zero_division=0
        )
        logger.info(f"Fold {fold} classification report:\n{report}")
        logger.info(f"Fold {fold} matriz de confusion (filas=real, cols=prediccion) [neutral, long, short]:\n{confusion_matrix(y_test, y_pred, labels=[0, 1, 2])}")
        if trade_stats["n_trades"] > 0:
            logger.info(
                f"Fold {fold} BACKTEST (comparable a walk_forward.py): {trade_stats['n_trades']} operaciones, "
                f"win_rate={trade_stats['win_rate'] * 100:.1f}%, pnl={trade_stats['total_pnl_pct'] * 100:.2f}%"
            )
            logger.info(
                f"Fold {fold} riesgo: sharpe={trade_stats['sharpe']:.3f}, sortino={trade_stats['sortino']:.3f}, "
                f"max_drawdown={trade_stats['max_drawdown_pct'] * 100:.2f}%"
            )
        else:
            logger.info(f"Fold {fold} BACKTEST: 0 operaciones")

        if args.fold_cooldown_seconds > 0 and fold < args.n_folds - 1:
            logger.info(f"Pausa de {args.fold_cooldown_seconds:.0f}s entre folds para disipar calor")
            time.sleep(args.fold_cooldown_seconds)

    logger.info("=== Resumen linea base (gradient boosting) ===")
    for i, (acc, maj) in enumerate(zip(fold_accuracies, fold_majority_accuracies), start=1):
        beats = "SUPERA" if acc > maj else "NO supera"
        logger.info(f"Fold {i}: accuracy={acc * 100:.1f}% vs baseline mayoritaria={maj * 100:.1f}% -> {beats}")
    logger.info(
        f"Accuracy media: {np.mean(fold_accuracies) * 100:.1f}% "
        f"(baseline mayoritaria media: {np.mean(fold_majority_accuracies) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
