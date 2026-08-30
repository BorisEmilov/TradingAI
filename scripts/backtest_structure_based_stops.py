"""Chequeo retrospectivo (solo analisis, no cambia entrenamiento ni ejecucion):
que pasaria si el SL/TP INICIAL se colocara en el nivel de estructura real mas
cercano (swing confirmado previo a la entrada) en vez del multiplo fijo de ATR
actual (`2xATR`/`4xATR`) -- ver conversacion 2026-08-29/30, opinion completa en
[[project-plan-2026-08-29]].

Mismo pipeline de señales walk-forward purgado que `backtest_structure_entry_filter.py`
(un solo ensemble por fold, SIN reentrenar nada nuevo -- esto es puramente sobre
donde se coloca el SL/TP de una señal ya generada). Para cada señal, compara DOS
variantes con la MISMA entrada/direccion:

- ATR (actual): SL = entry -/+ 2xATR, TP = entry +/- 4xATR, siempre.
- ESTRUCTURA (candidata): SL justo detras del ultimo swing confirmado PREVIO a la
  entrada (nunca mira el futuro), TP en el swing opuesto mas cercano ya
  establecido antes de la entrada. Se descarta la señal si no hay swing valido de
  algun lado, o si el R:R resultante no llega al piso de 1:2 -- en ambos casos se
  cuenta por separado (no se fuerza un TP/SL artificial).

Solo se comparan las señales donde la variante de estructura SI fue valida, para
que la comparacion sea A/B limpia sobre el mismo conjunto de operaciones.

Uso:
    python scripts/backtest_structure_based_stops.py --symbols EURUSD GBPJPY GBPAUD USDCHF
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
from tradingai.core.instruments import min_sl_distance_price  # noqa: E402
from tradingai.core.signal import MIN_RISK_REWARD_RATIO, Direction, TradingSignal  # noqa: E402
from tradingai.core.structure import confirmed_swing_highs, confirmed_swing_lows  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402
from tradingai.utils.thermal import wait_for_safe_temp  # noqa: E402

from loguru import logger  # noqa: E402

_DIRECTION_MAP = {0: Direction.NEUTRAL, 1: Direction.LONG, 2: Direction.SHORT}
_SL_BUFFER_ATR_MULT = 0.1  # margen mas alla del swing, para no quedar pegado exacto al nivel


def _structural_sl_tp(
    m15_candles, entry_idx: int, direction: Direction, entry_price: float, atr: float,
    swing_left: int, swing_right: int, lookback: int, symbol: str, min_sl_pips: float,
) -> tuple[float, float] | None:
    """SL/TP en el swing confirmado mas cercano PREVIO a la entrada -- nunca mira el
    futuro. None si no hay swing valido de algun lado o si no llega al piso de R:R.

    Aplica el MISMO piso minimo de distancia (`min_sl_distance_price`) que usa
    produccion para el SL basado en ATR -- sin esto, un swing muy cercano a la
    entrada da un SL de unos pocos pips, lo que infla el R:R artificialmente (un
    movimiento normal se ve como "20R" solo porque el denominador de riesgo es
    irrealmente chico, no porque la operacion sea buena de verdad). Se ensancha el
    SL al piso si hace falta, PERO el TP se deja en el nivel de estructura real sin
    tocar -- si eso ya no alcanza el piso de R:R, se descarta la señal en vez de
    forzar un TP artificial.
    """
    window = m15_candles.iloc[max(0, entry_idx - lookback + 1): entry_idx + 1]
    buffer = _SL_BUFFER_ATR_MULT * atr
    floor = min_sl_distance_price(symbol, min_sl_pips)

    if direction == Direction.LONG:
        lows = confirmed_swing_lows(window, swing_left, swing_right, count=1)
        highs = confirmed_swing_highs(window, swing_left, swing_right, count=1)
        if not lows or not highs:
            return None
        sl = min(lows[-1] - buffer, entry_price - floor)
        tp = highs[-1]
        if tp <= entry_price or sl >= entry_price:
            return None
    else:
        highs = confirmed_swing_highs(window, swing_left, swing_right, count=1)
        lows = confirmed_swing_lows(window, swing_left, swing_right, count=1)
        if not highs or not lows:
            return None
        sl = max(highs[-1] + buffer, entry_price + floor)
        tp = lows[-1]
        if tp >= entry_price or sl <= entry_price:
            return None

    risk = abs(entry_price - sl)
    reward = abs(tp - entry_price)
    if risk <= 0 or reward / risk < MIN_RISK_REWARD_RATIO:
        return None
    return sl, tp


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

    trades_atr, trades_structure = [], []
    n_sin_estructura = 0
    n_bajo_piso_rr = 0

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

        signals_atr, signals_structure = [], []
        for local_i in strided_local_indices:
            m15_row_idx = anchor_idx_test[local_i]
            direction_idx = int(y_pred[local_i])
            direction = _DIRECTION_MAP[direction_idx]
            if direction == Direction.NEUTRAL:
                continue
            confidence = float(y_proba[local_i, direction_idx])
            entry_price = float(m15_features["close"].iloc[m15_row_idx])
            atr = float(m15_features["atr_14"].iloc[m15_row_idx])
            if atr <= 0:
                continue

            sign = 1 if direction == Direction.LONG else -1
            tp_atr = entry_price + sign * tp_atr_mult * atr
            sl_atr = entry_price - sign * sl_atr_mult * atr

            structural = _structural_sl_tp(
                m15_candles, m15_row_idx, direction, entry_price, atr,
                args.swing_left, args.swing_right, args.swing_lookback,
                symbol, args.min_sl_pips,
            )
            if structural is None:
                n_sin_estructura += 1
                continue
            sl_struct, tp_struct = structural
            if abs(tp_struct - entry_price) / abs(entry_price - sl_struct) < MIN_RISK_REWARD_RATIO:
                n_bajo_piso_rr += 1
                continue

            base_kwargs = dict(
                symbol=symbol, timeframe="M15", timestamp=datetime.now(timezone.utc),
                direction=direction, confidence=confidence, entry_price=entry_price,
            )
            signals_atr.append((m15_row_idx, TradingSignal(**base_kwargs, stop_loss=sl_atr, take_profit=tp_atr)))
            signals_structure.append((m15_row_idx, TradingSignal(**base_kwargs, stop_loss=sl_struct, take_profit=tp_struct)))

        backtest_cfg = config.get("backtest", {})
        backtester = Backtester(
            confidence_threshold=args.confidence_threshold,
            max_holding_bars=backtest_cfg.get("max_holding_bars", 50),
            spread_pips=resolve_spread_pips(backtest_cfg, symbol),
            slippage_pips=backtest_cfg.get("slippage_pips", 0.2),
            commission_pips=backtest_cfg.get("commission_pips", 0.0),
            pip_size=pip_size(symbol),
        )
        trades_atr.extend(backtester.run(m15_candles, signals_atr))
        trades_structure.extend(backtester.run(m15_candles, signals_structure))

        logger.info(
            f"[{symbol}] fold {fold}/{args.n_folds - 1} entrenado en {train_time:.1f}s, "
            f"{len(signals_structure)} señales con estructura valida"
        )
        if args.fold_cooldown_seconds > 0 and fold < args.n_folds - 1:
            time.sleep(args.fold_cooldown_seconds)

    return summarize(trades_atr), summarize(trades_structure), n_sin_estructura, n_bajo_piso_rr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--backtest-step", type=int, default=10)
    parser.add_argument("--swing-left", type=int, default=3)
    parser.add_argument("--swing-right", type=int, default=3)
    parser.add_argument("--swing-lookback", type=int, default=100)
    parser.add_argument("--min-sl-pips", type=float, default=12.0, help="Mismo piso que usa produccion (core.instruments)")
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

    totals_atr, totals_structure = [], []
    total_sin_estructura = total_bajo_piso = 0

    for symbol in args.symbols:
        logger.info(f"=== {symbol} ===")
        stats_atr, stats_structure, n_sin_estructura, n_bajo_piso = _run_symbol(symbol, args, config)
        totals_atr.append(stats_atr)
        totals_structure.append(stats_structure)
        total_sin_estructura += n_sin_estructura
        total_bajo_piso += n_bajo_piso

        logger.info(
            f"[{symbol}] ATR:        n={stats_atr['n_trades']} win_rate={stats_atr.get('win_rate', 0)*100:.1f}% "
            f"expectancy_r={stats_atr.get('expectancy_r', 0):.3f} profit_factor={stats_atr.get('profit_factor', 0):.3f}"
        )
        logger.info(
            f"[{symbol}] ESTRUCTURA: n={stats_structure['n_trades']} win_rate={stats_structure.get('win_rate', 0)*100:.1f}% "
            f"expectancy_r={stats_structure.get('expectancy_r', 0):.3f} profit_factor={stats_structure.get('profit_factor', 0):.3f}"
        )

    n_a = sum(s["n_trades"] for s in totals_atr)
    n_s = sum(s["n_trades"] for s in totals_structure)
    er_a = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_atr) / n_a if n_a else 0.0
    er_s = sum(s.get("expectancy_r", 0) * s["n_trades"] for s in totals_structure) / n_s if n_s else 0.0

    logger.info("=== TOTAL AGREGADO ===")
    logger.info(f"ATR:        n={n_a} expectancy_r_promedio={er_a:.3f}")
    logger.info(f"ESTRUCTURA: n={n_s} expectancy_r_promedio={er_s:.3f}")
    logger.info(
        f"Señales descartadas -- sin swing valido de algun lado: {total_sin_estructura}, "
        f"con estructura pero por debajo del piso 1:2 de R:R: {total_bajo_piso}"
    )


if __name__ == "__main__":
    main()
