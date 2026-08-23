"""Backtest simplificado de un checkpoint entrenado sobre historico D1/H1/M15/M5.

Por defecto evalua **solo el tramo de validacion** (el mismo split cronologico 80/20
que uso train.py) para no mezclar datos que el modelo ya vio en entrenamiento con la
medicion de calidad — usar --full-history para el barrido completo (util para debug,
pero optimista como medida de calidad real).

Recalcula features desde cero en cada anchor M15 (sin cache), pero sobre una ventana
acotada por temporalidad (no todo el historico) para que el costo sea O(n) total y no
O(n^2). Sigue siendo mas lento que un backtest vectorizado; para datasets grandes, sube
--step.

Uso:
    python scripts/backtest.py --checkpoint data/models/EURUSD_transformer.pt --symbol EURUSD
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

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, last_closed_bar_index  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.evaluation.backtester import Backtester, summarize  # noqa: E402
from tradingai.ai.inference.gbm_predictor import GBMPredictor  # noqa: E402
from tradingai.ai.inference.predictor import Predictor  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.core.pipeline import FEATURE_WARMUP_BARS  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402

from loguru import logger  # noqa: E402


def _pip_size(symbol: str) -> float:
    """Heuristico offline (sin conexion a MT5): JPY y metales cotizan con pip mas grande."""
    if symbol.upper().endswith("JPY"):
        return 0.01
    if symbol.upper() in {"XAUUSD", "XAGUSD"}:
        return 0.01
    return 0.0001


def _validation_start_index(
    candles_by_tf: dict, feature_columns: list[str], seq_len_by_tf: dict[str, int], val_split: float, config: dict
) -> int:
    """Indice (en candles_by_tf["M15"]) donde empieza el 20% de validacion usado en train.py.

    Reconstruye el mismo dataset alineado que uso el entrenamiento (mismo split
    cronologico 80/20, sin shuffle) para saber, en velas M15 crudas, donde arranca el
    tramo que el modelo nunca vio.
    """
    features_by_tf = {
        tf: normalize_ohlcv(build_feature_pipeline(candles_by_tf[tf], config)) for tf in TIMEFRAMES
    }
    dataset = MultiTimeframeTradingDataset(features_by_tf, feature_columns, seq_len_by_tf)
    split_idx = int(len(dataset) * (1 - val_split))

    val_start_ts = features_by_tf["M15"]["timestamp"].iloc[dataset.anchor_positions[split_idx]]
    m15_ts = candles_by_tf["M15"]["timestamp"].to_numpy()
    return int(np.searchsorted(m15_ts, val_start_ts.to_numpy(), side="left"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-dir", default=None, help="Por defecto, paths.raw_data de config.yaml")
    parser.add_argument("--step", type=int, default=5, help="cada cuantos anchors M15 se genera una senal")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Por defecto, el de config.yaml. Bajarlo aqui es solo para evaluacion, no cambia produccion.",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Evaluar sobre todo el historico en vez de solo el tramo de validacion (optimista, solo para debug).",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Debe coincidir con el usado en train.py")
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="Ignorar spread/slippage/comision (para comparar contra el resultado con costes).",
    )
    args = parser.parse_args()

    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    data_dir = Path(args.data_dir or config["paths"]["raw_data"])
    candles_by_tf = {tf: load_csv(data_dir / f"{args.symbol}_{tf}.csv") for tf in TIMEFRAMES}

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.suffix == ".joblib":
        predictor = GBMPredictor.from_checkpoint(checkpoint_path)
    else:
        feature_columns = torch.load(checkpoint_path, map_location="cpu")["feature_columns"]
        predictor = Predictor.from_checkpoint(checkpoint_path, feature_columns)
    feature_columns = predictor.feature_columns

    # Ventana acotada por temporalidad: suficiente para seq_len + warm-up de indicadores,
    # sin arrastrar todo el historico en cada iteracion (séria O(n^2)).
    max_window = {tf: seq_len + FEATURE_WARMUP_BARS for tf, seq_len in predictor.seq_len_by_tf.items()}

    m15 = candles_by_tf["M15"]
    anchor_ts = m15["timestamp"].to_numpy()
    cutoff_idx = {
        tf: last_closed_bar_index(candles_by_tf[tf]["timestamp"].to_numpy(), anchor_ts) for tf in TIMEFRAMES
    }

    start_idx = 0
    if not args.full_history:
        logger.info("Calculando el inicio del tramo de validacion (mismo split que train.py)...")
        start_idx = _validation_start_index(
            candles_by_tf, feature_columns, predictor.seq_len_by_tf, args.val_split, config
        )
        logger.info(
            f"Evaluando solo sobre validacion: desde {m15['timestamp'].iloc[start_idx]} "
            f"({len(m15) - start_idx} de {len(m15)} velas M15). Usa --full-history para el barrido completo."
        )

    anchor_indices = list(range(start_idx, len(m15), args.step))
    signals = []
    n_no_history = 0
    n_prediction_failed = 0
    t0 = time.time()
    for n, i in enumerate(anchor_indices, start=1):
        if any(cutoff_idx[tf][i] < 0 for tf in TIMEFRAMES):
            n_no_history += 1
        else:
            window_by_tf = {
                tf: candles_by_tf[tf].iloc[max(0, cutoff_idx[tf][i] + 1 - max_window[tf]) : cutoff_idx[tf][i] + 1]
                for tf in TIMEFRAMES
            }
            try:
                signals.append((i, predictor.predict(window_by_tf, symbol=args.symbol)))
            except ValueError as exc:
                n_prediction_failed += 1
                if n_prediction_failed == 1:
                    logger.debug(f"Primer fallo de prediccion (anchor {i}): {exc}")

        if n % 20 == 0 or n == len(anchor_indices):
            logger.info(
                f"{n}/{len(anchor_indices)} anchors procesados ({time.time() - t0:.1f}s) - "
                f"{len(signals)} señales, {n_no_history} sin historia, {n_prediction_failed} fallidas"
            )

    logger.info(f"{len(signals)} señales generadas sobre {len(m15)} velas M15")

    if signals:
        confs = np.array([sig.confidence for _, sig in signals])
        logger.info(
            f"confianza: min={confs.min():.3f} p50={np.percentile(confs, 50):.3f} "
            f"p90={np.percentile(confs, 90):.3f} max={confs.max():.3f}"
        )

    threshold = args.confidence_threshold
    if threshold is None:
        threshold = config["model"]["outputs"]["confidence_threshold"]
    logger.info(f"umbral de confianza usado: {threshold}")

    backtest_cfg = config.get("backtest", {})
    if args.no_costs:
        spread_pips = slippage_pips = commission_pips = 0.0
    else:
        spread_pips = backtest_cfg.get("spread_pips", 1.0)
        slippage_pips = backtest_cfg.get("slippage_pips", 0.2)
        commission_pips = backtest_cfg.get("commission_pips", 0.0)
    logger.info(
        f"costes: spread={spread_pips} slippage={slippage_pips} comision={commission_pips} pips "
        f"(pip_size={_pip_size(args.symbol)})"
    )

    backtester = Backtester(
        confidence_threshold=threshold,
        max_holding_bars=backtest_cfg.get("max_holding_bars", 50),
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_pips=commission_pips,
        pip_size=_pip_size(args.symbol),
    )
    trades = backtester.run(m15, signals)

    stats = summarize(trades)
    if stats["n_trades"] == 0:
        logger.info("0 operaciones (ninguna senal supero el umbral de confianza)")
    else:
        logger.info(
            f"{stats['n_trades']} operaciones - win_rate={stats['win_rate'] * 100:.1f}% - "
            f"pnl medio={stats['avg_pnl_pct'] * 100:.3f}% - pnl total={stats['total_pnl_pct'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
