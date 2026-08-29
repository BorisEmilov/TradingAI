"""Evalua si "setups" combinados de accion de precio (varias condiciones ICT/SMC
confluyendo a la vez, no un solo feature aislado) predicen mejores operaciones que
el resto -- ver conversacion 2026-08-29: un filtro de UN solo feature (estructura
BOS/CHoCH) ya se probo y no dio beneficio (`backtest_structure_entry_filter.py`,
0.455 vs 0.445 expectancy_r). Esto prueba una hipotesis distinta: quizas la
COMBINACION de condiciones si tenga señal aunque cada una sola no la tenga.

Reutiliza el mismo pipeline de señales que `backtest_structure_entry_filter.py`
(entrena el mismo ensemble por fold purgado, genera trades ESTATICOS con TP/SL
fijo), pero en vez de partir por un solo flag de estructura, parte por 3 combos
de condiciones ya calculadas por `build_feature_pipeline` (no hace falta
recalcular nada, ya son columnas en `features_by_tf["M15"]`):

- Combo A (OB+FVG): order block y FVG a favor de la direccion de la señal.
- Combo B (Turtle Soup): barrida de liquidez dentro de una killzone ICT.
- Combo C (OTE + tendencia): retroceso en zona dorada de Fibonacci con la
  tendencia de estructura confirmando la misma direccion.

Uso:
    python scripts/backtest_setup_combos.py --symbols EURUSD GBPJPY GBPAUD USDCHF
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

COMBOS = ("ob_fvg", "turtle_soup", "ote_trend")


def _combo_membership(m15_features, row_idx: int, direction: Direction) -> dict[str, bool]:
    """Evalua los 3 combos EN EL MOMENTO de la señal (misma vela de entrada)."""
    row = m15_features.iloc[row_idx]
    is_long = direction == Direction.LONG

    ob_fvg = (row["bullish_ob"] and row["bullish_fvg"]) if is_long else (row["bearish_ob"] and row["bearish_fvg"])

    turtle_soup = bool(
        row["killzone_liquidity_sweep_bullish"] if is_long else row["killzone_liquidity_sweep_bearish"]
    )

    ote_trend = (bool(row["in_ote_bullish"]) and row["trend"] == 1) if is_long else (
        bool(row["in_ote_bearish"]) and row["trend"] == -1
    )

    return {"ob_fvg": bool(ob_fvg), "turtle_soup": turtle_soup, "ote_trend": bool(ote_trend)}


def _run_symbol(symbol: str, args: argparse.Namespace, config: dict) -> dict:
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

    trades_present = {c: [] for c in COMBOS}
    trades_absent = {c: [] for c in COMBOS}

    for fold in range(1, args.n_folds):
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
        membership_by_idx = {}
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
                membership_by_idx[m15_row_idx] = _combo_membership(m15_features, m15_row_idx, direction)

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

        signal_by_key = {(s.entry_price, s.direction): idx for idx, s in signals if s.direction != Direction.NEUTRAL}
        for t in trades:
            key = (t.signal.entry_price, t.signal.direction)
            idx = signal_by_key.get(key)
            membership = membership_by_idx.get(idx, {})
            for combo in COMBOS:
                if membership.get(combo, False):
                    trades_present[combo].append(t)
                else:
                    trades_absent[combo].append(t)

        logger.info(f"[{symbol}] fold {fold}/{args.n_folds - 1} entrenado en {train_time:.1f}s")
        if args.fold_cooldown_seconds > 0 and fold < args.n_folds - 1:
            time.sleep(args.fold_cooldown_seconds)

    return {
        combo: (summarize(trades_present[combo]), summarize(trades_absent[combo]))
        for combo in COMBOS
    }


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

    totals = {combo: {"present": [], "absent": []} for combo in COMBOS}

    for symbol in args.symbols:
        logger.info(f"=== {symbol} ===")
        results = _run_symbol(symbol, args, config)
        for combo in COMBOS:
            present_stats, absent_stats = results[combo]
            totals[combo]["present"].append(present_stats)
            totals[combo]["absent"].append(absent_stats)

            n_p, n_a = present_stats["n_trades"], absent_stats["n_trades"]
            pct_present = n_p / (n_p + n_a) * 100 if (n_p + n_a) else 0.0
            logger.info(
                f"[{symbol}] {combo} PRESENTE: n={n_p} win_rate={present_stats.get('win_rate', 0)*100:.1f}% "
                f"expectancy_r={present_stats.get('expectancy_r', 0):.3f} "
                f"profit_factor={present_stats.get('profit_factor', 0):.3f} ({pct_present:.1f}% de las señales)"
            )
            logger.info(
                f"[{symbol}] {combo} AUSENTE:   n={n_a} win_rate={absent_stats.get('win_rate', 0)*100:.1f}% "
                f"expectancy_r={absent_stats.get('expectancy_r', 0):.3f} "
                f"profit_factor={absent_stats.get('profit_factor', 0):.3f}"
            )

    logger.info("=== TOTAL AGREGADO ===")
    for combo in COMBOS:
        stats_p = totals[combo]["present"]
        stats_a = totals[combo]["absent"]
        n_p = sum(s["n_trades"] for s in stats_p)
        n_a = sum(s["n_trades"] for s in stats_a)
        er_p = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in stats_p) / n_p if n_p else 0.0
        er_a = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in stats_a) / n_a if n_a else 0.0
        logger.info(f"{combo}: PRESENTE n={n_p} expectancy_r_promedio={er_p:.3f} | AUSENTE n={n_a} expectancy_r_promedio={er_a:.3f}")


if __name__ == "__main__":
    main()
