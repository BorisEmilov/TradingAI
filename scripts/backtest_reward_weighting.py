"""Valida si ponderar el entrenamiento por la magnitud del R realizado (en vez de
peso uniforme, el actual) mejora expectancy_r -- ver conversacion 2026-08-29.

La etiqueta de produccion (`triple_barrier_labels`) es categorica (0=neutral,
1=long, 2=short): tira a la basura CUANTO se movio el precio, solo importa si
"gano" o "perdio" la barrera. Un diagnostico previo (histograma de R realizado
sobre ~90k ejemplos de EURUSD/GBPJPY) mostro variacion real y continua, no un
escalon binario -- justifica probar si darle mas peso en el entrenamiento a los
ejemplos con movimiento mas grande/claro (y menos a los marginales) mejora el
resultado real.

Compara, con el MISMO pipeline de señales walk-forward purgado que
`backtest_structure_entry_filter.py`: un ensemble entrenado con sample_weight
uniforme (=1, el actual) contra uno entrenado con sample_weight = |R realizado|
(acotado para no dejar que un outlier domine), sobre trades ESTATICOS (TP/SL
fijo, para aislar el efecto de la ponderacion de la gestion dinamica de salida).

Uso:
    python scripts/backtest_reward_weighting.py --symbols EURUSD GBPJPY GBPAUD USDCHF
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
_WEIGHT_CLIP_MIN = 0.2
_WEIGHT_CLIP_MAX = 3.0


def _realized_r_multiples(df, horizon: int, tp_atr_mult: float, sl_atr_mult: float) -> np.ndarray:
    """Cuanto se movio el precio realmente en R (no solo si toco la barrera nominal):
    para barreras tocadas, el maximo/minimo alcanzado ANTES del cierre (overshoot real);
    para timeouts, el movimiento al final del horizonte. Solo para ponderar el
    entrenamiento -- NO reemplaza `triple_barrier_labels` (la clase categorica sigue
    siendo la etiqueta real, esto solo pesa cuanto importa cada ejemplo)."""
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr_14"].to_numpy()
    n = len(df)
    r = np.zeros(n, dtype=np.float64)

    for i in range(n - horizon):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        entry = close[i]
        sl_dist = sl_atr_mult * atr[i]
        upper = entry + tp_atr_mult * atr[i]
        lower = entry - sl_dist

        future_high = high[i + 1 : i + 1 + horizon]
        future_low = low[i + 1 : i + 1 + horizon]
        hit_upper_idx = np.argmax(future_high >= upper) if (future_high >= upper).any() else None
        hit_lower_idx = np.argmax(future_low <= lower) if (future_low <= lower).any() else None

        if hit_upper_idx is not None and (hit_lower_idx is None or hit_upper_idx <= hit_lower_idx):
            r[i] = (future_high[: hit_upper_idx + 1].max() - entry) / sl_dist
        elif hit_lower_idx is not None:
            r[i] = (future_low[: hit_lower_idx + 1].min() - entry) / sl_dist
        else:
            r[i] = (close[i + horizon] - entry) / sl_dist

    return r


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

    m15_features = features_by_tf["M15"]
    realized_r = _realized_r_multiples(m15_features, dataset.horizon, tp_atr_mult, sl_atr_mult)
    weights_full = np.clip(np.abs(realized_r), _WEIGHT_CLIP_MIN, _WEIGHT_CLIP_MAX)
    w = weights_full[dataset.anchor_positions]

    fold_bounds = [int(n * i / args.n_folds) for i in range(args.n_folds + 1)]
    trades_uniform, trades_weighted = [], []

    for fold in range(1, args.n_folds):
        test_start, test_end = fold_bounds[fold], fold_bounds[fold + 1]
        purged_train_end = test_start - dataset.horizon
        X_train, y_train, w_train = X[:purged_train_end], y[:purged_train_end], w[:purged_train_end]
        X_test = X[test_start:test_end]

        for variant, sample_weight in (("uniform", None), ("weighted", w_train)):
            t0 = time.time()
            models = []
            with threadpoolctl.threadpool_limits(limits=args.max_threads):
                for i in range(args.n_seeds):
                    wait_for_safe_temp(max_temp_c=args.max_temp_c)
                    m = HistGradientBoostingClassifier(random_state=args.seed + i, max_iter=200, early_stopping=True)
                    m.fit(X_train, y_train, sample_weight=sample_weight)
                    models.append(m)
            train_time = time.time() - t0

            y_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
            y_pred = y_proba.argmax(axis=1)

            m15_candles = candles_by_tf["M15"]
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
            backtester = Backtester(
                confidence_threshold=args.confidence_threshold,
                max_holding_bars=backtest_cfg.get("max_holding_bars", 50),
                spread_pips=resolve_spread_pips(backtest_cfg, symbol),
                slippage_pips=backtest_cfg.get("slippage_pips", 0.2),
                commission_pips=backtest_cfg.get("commission_pips", 0.0),
                pip_size=pip_size(symbol),
            )
            trades = backtester.run(m15_candles, signals)
            (trades_uniform if variant == "uniform" else trades_weighted).extend(trades)

            logger.info(f"[{symbol}] fold {fold}/{args.n_folds - 1} [{variant}] entrenado en {train_time:.1f}s, {len(trades)} trades")

        if args.fold_cooldown_seconds > 0 and fold < args.n_folds - 1:
            time.sleep(args.fold_cooldown_seconds)

    return summarize(trades_uniform), summarize(trades_weighted)


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
    parser.add_argument("--max-temp-c", type=float, default=90.0)
    parser.add_argument("--fold-cooldown-seconds", type=float, default=20.0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    totals_uniform, totals_weighted = [], []
    for symbol in args.symbols:
        logger.info(f"=== {symbol} ===")
        stats_uniform, stats_weighted = _run_symbol(symbol, args, config)
        totals_uniform.append(stats_uniform)
        totals_weighted.append(stats_weighted)
        logger.info(
            f"[{symbol}] UNIFORME:   n={stats_uniform['n_trades']} win_rate={stats_uniform.get('win_rate', 0)*100:.1f}% "
            f"expectancy_r={stats_uniform.get('expectancy_r', 0):.3f} profit_factor={stats_uniform.get('profit_factor', 0):.3f}"
        )
        logger.info(
            f"[{symbol}] PONDERADO:  n={stats_weighted['n_trades']} win_rate={stats_weighted.get('win_rate', 0)*100:.1f}% "
            f"expectancy_r={stats_weighted.get('expectancy_r', 0):.3f} profit_factor={stats_weighted.get('profit_factor', 0):.3f}"
        )

    n_u = sum(s["n_trades"] for s in totals_uniform)
    n_w = sum(s["n_trades"] for s in totals_weighted)
    er_u = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_uniform) / n_u if n_u else 0.0
    er_w = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_weighted) / n_w if n_w else 0.0

    logger.info("=== TOTAL AGREGADO ===")
    logger.info(f"UNIFORME:  n={n_u} expectancy_r_promedio={er_u:.3f}")
    logger.info(f"PONDERADO: n={n_w} expectancy_r_promedio={er_w:.3f}")


if __name__ == "__main__":
    main()
