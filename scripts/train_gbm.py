"""Entrena el modelo real de produccion: gradient boosting sobre el snapshot mas
reciente de cada temporalidad (ver src/tradingai/ai/inference/gbm_predictor.py).

Entrena sobre TODO el historico disponible del simbolo — la validacion de que
generaliza ya se hizo con walk-forward (scripts/baseline_gbm.py, ver memoria del
proyecto: 20/20 folds positivos en 5 simbolos). El objetivo aqui es maximizar lo que
el modelo de produccion aprende, no volver a validar.

Uso:
    python scripts/train_gbm.py --symbol EURUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, flatten_last_timestep  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402

from loguru import logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-dir", default=None, help="Por defecto, paths.raw_data de config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    data_dir = Path(args.data_dir or config["paths"]["raw_data"])

    features_by_tf = {}
    for tf in TIMEFRAMES:
        csv_path = data_dir / f"{args.symbol}_{tf}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No se encontro {csv_path}. Genera el historico con: "
                f"python scripts/fetch_historical_data.py --symbols {args.symbol}"
            )
        candles = load_csv(csv_path)
        features_by_tf[tf] = normalize_ohlcv(build_feature_pipeline(candles, config))
        logger.info(f"{tf}: {len(features_by_tf[tf])} velas con features validas ({csv_path.name})")

    feature_columns = select_feature_columns(features_by_tf["M15"])
    seq_len_by_tf = config["model"]["sequence_length"]
    dataset = MultiTimeframeTradingDataset(features_by_tf, feature_columns, seq_len_by_tf)
    logger.info(f"Dataset: {len(dataset)} ejemplos alineados en las 4 temporalidades")

    X = flatten_last_timestep(dataset.sequences)
    y = dataset.direction
    logger.info(f"X shape: {X.shape}, distribucion direction: {np.bincount(y, minlength=3).tolist()}")

    model = HistGradientBoostingClassifier(random_state=args.seed, max_iter=200, early_stopping=True)
    model.fit(X, y)
    logger.info("Modelo entrenado sobre todo el historico disponible")

    # "secrets" (credenciales/URL del bridge) no hace falta para inferencia y no es un
    # tipo de datos puro (pydantic BaseSettings) — no se guarda en el checkpoint.
    config_without_secrets = {k: v for k, v in config.items() if k != "secrets"}

    out_path = Path(config["paths"]["models_dir"]) / f"{args.symbol}_gbm.joblib"
    joblib.dump(
        {"model": model, "feature_columns": feature_columns, "config": config_without_secrets},
        out_path,
    )
    logger.info(f"Checkpoint guardado en {out_path}")


if __name__ == "__main__":
    main()
