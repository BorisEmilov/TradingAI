"""Entrena el modelo real de produccion: gradient boosting sobre el snapshot mas
reciente de cada temporalidad (ver src/tradingai/ai/inference/gbm_predictor.py).

Entrena sobre TODO el historico disponible del simbolo — la validacion de que
generaliza ya se hizo con walk-forward (scripts/baseline_gbm.py, ver memoria del
proyecto: 20/20 folds positivos en 5 simbolos). El objetivo aqui es maximizar lo que
el modelo de produccion aprende, no volver a validar.

Entrena un ENSEMBLE de `--n-seeds` modelos con semillas distintas (mismos datos,
misma arquitectura) y los guarda todos juntos -- `GBMPredictor` promedia sus
probabilidades en inferencia. Reduce la varianza de "le toco una inicializacion
buena/mala" que ya se vio con el transformer (mismo "mejor epoch" daba resultados
muy distintos entre corridas por pura suerte de semilla, ver memoria del proyecto).

Uso:
    python scripts/train_gbm.py --symbol EURUSD
    python scripts/train_gbm.py --symbol EURUSD --n-seeds 5
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
import threadpoolctl  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, flatten_last_timestep  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402
from tradingai.utils.thermal import wait_for_safe_temp  # noqa: E402

from loguru import logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-dir", default=None, help="Por defecto, paths.raw_data de config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base; el ensemble usa seed, seed+1, ... seed+n_seeds-1")
    parser.add_argument("--n-seeds", type=int, default=5, help="Numero de modelos en el ensemble")
    # Bug real encontrado el 2026-08-29: este script entrenaba sin ningun limite de
    # hilos ni vigilancia termica (49 hilos BLAS observados en vivo, CPU sostenida a
    # 99-101C durante el reentreno semanal) -- el mismo patron que causo el
    # load1=223 anomalo del 2026-08-24 en inferencia (ver gbm_predictor.py), nunca
    # corregido aca porque este script se usa menos seguido. baseline_gbm.py (la
    # validacion walk-forward) SI tenia esta proteccion desde el principio.
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument("--max-temp-c", type=float, default=80.0)
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
    dataset = MultiTimeframeTradingDataset(
        features_by_tf, feature_columns, seq_len_by_tf,
        tp_atr_mult=config["model"]["tp_atr_mult"], sl_atr_mult=config["model"]["sl_atr_mult"],
    )
    logger.info(f"Dataset: {len(dataset)} ejemplos alineados en las 4 temporalidades")

    X = flatten_last_timestep(dataset.sequences)
    y = dataset.direction
    logger.info(f"X shape: {X.shape}, distribucion direction: {np.bincount(y, minlength=3).tolist()}")

    models = []
    with threadpoolctl.threadpool_limits(limits=args.max_threads):
        for i in range(args.n_seeds):
            wait_for_safe_temp(max_temp_c=args.max_temp_c)
            seed = args.seed + i
            model = HistGradientBoostingClassifier(random_state=seed, max_iter=200, early_stopping=True)
            model.fit(X, y)
            models.append(model)
            logger.info(f"Modelo {i + 1}/{args.n_seeds} entrenado (seed={seed})")

    # "secrets" (credenciales/URL del bridge) no hace falta para inferencia y no es un
    # tipo de datos puro (pydantic BaseSettings) — no se guarda en el checkpoint.
    config_without_secrets = {k: v for k, v in config.items() if k != "secrets"}

    out_path = Path(config["paths"]["models_dir"]) / f"{args.symbol}_gbm.joblib"
    joblib.dump(
        {"models": models, "feature_columns": feature_columns, "config": config_without_secrets},
        out_path,
    )
    logger.info(f"Checkpoint guardado en {out_path} (ensemble de {len(models)} modelos)")


if __name__ == "__main__":
    main()
