"""Evalua retroactivamente si filtrar ENTRADAS por estructura (en vez de solo
gestionar salidas) habria mejorado el resultado -- ver conversacion 2026-08-28: el
usuario noto que muchas operaciones se abren y cierran casi enseguida por
invalidacion de estructura porque la estructura YA estaba rota al momento de abrir,
no porque se rompiera despues.

Reutiliza el MISMO pipeline de señales que scripts/backtest_structure_exit.py
(entrena el mismo ensemble por fold, genera las mismas señales), pero en vez de
simular una gestion de salida dinamica, parte las operaciones ESTATICAS (TP/SL
fijo, sin exits dinamicos) en dos grupos segun si la estructura ya estaba a favor
o en contra en el momento exacto de la señal, y compara expectancy_r/profit_factor
entre los dos grupos.

Uso:
    python scripts/backtest_structure_entry_filter.py --symbols EURUSD GBPJPY GBPAUD USDCHF
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
from tradingai.core.structure import confirmed_swing_highs, confirmed_swing_lows  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402
from tradingai.utils.thermal import wait_for_safe_temp  # noqa: E402

from loguru import logger  # noqa: E402

_DIRECTION_MAP = {0: Direction.NEUTRAL, 1: Direction.LONG, 2: Direction.SHORT}


def _structure_disagrees_at_entry(
    candles, idx: int, direction: Direction, swing_left: int = 3, swing_right: int = 3, lookback: int = 100
) -> bool:
    """Misma logica que mt5.structure_exit.structure_invalidated, pero evaluada
    EN EL MOMENTO DE LA SEÑAL (hasta el indice `idx` inclusive) en vez de sobre una
    posicion ya abierta -- esto es lo que un filtro de entrada rechazaria."""
    window = candles.iloc[max(0, idx - lookback + 1): idx + 1]
    if direction == Direction.LONG:
        lows = confirmed_swing_lows(window, swing_left, swing_right, count=2)
        return len(lows) >= 2 and lows[-1] < lows[-2]
    if direction == Direction.SHORT:
        highs = confirmed_swing_highs(window, swing_left, swing_right, count=2)
        return len(highs) >= 2 and highs[-1] > highs[-2]
    return False


def _run_symbol(symbol: str, args: argparse.Namespace, config: dict) -> tuple[dict, dict, int, int]:
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

    agree_trades = []
    disagree_trades = []
    n_agree_signals = 0
    n_disagree_signals = 0

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
        m15_candles = candles_by_tf["M15"]
        anchor_idx_test = dataset.anchor_positions[test_start:test_end]
        strided_local_indices = range(0, len(anchor_idx_test), args.backtest_step)

        signals = []
        structure_flags = {}
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

            signal = TradingSignal(
                symbol=symbol, timeframe="M15", timestamp=datetime.now(timezone.utc),
                direction=direction, confidence=confidence, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
            )
            signals.append((m15_row_idx, signal))

            if direction in (Direction.LONG, Direction.SHORT):
                disagrees = _structure_disagrees_at_entry(m15_candles, m15_row_idx, direction)
                structure_flags[m15_row_idx] = disagrees

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

        # Backtester.run() no devuelve el idx de cada trade -- reconstruimos el
        # emparejamiento por entry_price+direction, unico dentro de un fold dado el
        # espaciado de anchors (backtest_step evita entries superpuestas).
        signal_by_key = {(s.entry_price, s.direction): idx for idx, s in signals if s.direction != Direction.NEUTRAL}
        for t in trades:
            key = (t.signal.entry_price, t.signal.direction)
            idx = signal_by_key.get(key)
            disagrees = structure_flags.get(idx, False)
            if disagrees:
                disagree_trades.append(t)
                n_disagree_signals += 1
            else:
                agree_trades.append(t)
                n_agree_signals += 1

        logger.info(f"[{symbol}] fold {fold}/{args.n_folds - 1} entrenado en {train_time:.1f}s")
        if args.fold_cooldown_seconds > 0 and fold < args.n_folds - 1:
            time.sleep(args.fold_cooldown_seconds)

    return summarize(agree_trades), summarize(disagree_trades), n_agree_signals, n_disagree_signals


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
    parser.add_argument("--max-temp-c", type=float, default=78.0)
    parser.add_argument("--fold-cooldown-seconds", type=float, default=20.0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    totals_agree, totals_disagree = [], []
    for symbol in args.symbols:
        logger.info(f"=== {symbol} ===")
        agree_stats, disagree_stats, n_agree, n_disagree = _run_symbol(symbol, args, config)
        totals_agree.append(agree_stats)
        totals_disagree.append(disagree_stats)

        pct_rejected = n_disagree / (n_agree + n_disagree) * 100 if (n_agree + n_disagree) else 0.0
        logger.info(
            f"[{symbol}] ESTRUCTURA A FAVOR: n={agree_stats['n_trades']} win_rate={agree_stats.get('win_rate', 0)*100:.1f}% "
            f"expectancy_r={agree_stats.get('expectancy_r', 0):.3f} profit_factor={agree_stats.get('profit_factor', 0):.3f}"
        )
        logger.info(
            f"[{symbol}] ESTRUCTURA EN CONTRA (rechazadas por el filtro): n={disagree_stats['n_trades']} "
            f"win_rate={disagree_stats.get('win_rate', 0)*100:.1f}% expectancy_r={disagree_stats.get('expectancy_r', 0):.3f} "
            f"profit_factor={disagree_stats.get('profit_factor', 0):.3f} ({pct_rejected:.1f}% de las señales se habrian rechazado)"
        )

    n_a = sum(s["n_trades"] for s in totals_agree)
    n_d = sum(s["n_trades"] for s in totals_disagree)
    er_a = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_agree) / n_a if n_a else 0.0
    er_d = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_disagree) / n_d if n_d else 0.0

    logger.info("=== TOTAL AGREGADO ===")
    logger.info(f"ESTRUCTURA A FAVOR:    n={n_a} expectancy_r_promedio={er_a:.3f}")
    logger.info(f"ESTRUCTURA EN CONTRA:  n={n_d} expectancy_r_promedio={er_d:.3f} (esto es lo que un filtro de entrada eliminaria)")


if __name__ == "__main__":
    main()
