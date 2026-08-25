"""Entrena el modelo multi-temporalidad a partir de historico D1/H1/M15/M5.

Espera los CSVs generados por scripts/fetch_historical_data.py:
    data/raw/{SYMBOL}_D1.csv, {SYMBOL}_H1.csv, {SYMBOL}_M15.csv, {SYMBOL}_M5.csv

Con varios --symbols, cada simbolo se alinea y se divide train/val por separado
(80/20 cronologico propio, sin mezclar simbolos en el split) y luego se concatenan —
asi el modelo ve mas variedad sin que el split de validacion de un simbolo se filtre
con datos de otro.

Uso:
    python scripts/train.py --symbols EURUSD
    python scripts/train.py --symbols EURUSD GBPUSD XAUUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.models.base import build_model  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.ai.training.trainer import Trainer  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402
from tradingai.utils.seed import set_seed  # noqa: E402

from loguru import logger  # noqa: E402
from torch.utils.data import ConcatDataset, Subset  # noqa: E402


def _build_dataset_for_symbol(
    symbol: str, data_dir: Path, config: dict
) -> tuple[MultiTimeframeTradingDataset, list[str]]:
    features_by_tf = {}
    for tf in TIMEFRAMES:
        csv_path = data_dir / f"{symbol}_{tf}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No se encontro {csv_path}. Genera el historico con: "
                f"python scripts/fetch_historical_data.py --symbols {symbol}"
            )
        candles = load_csv(csv_path)
        features = normalize_ohlcv(build_feature_pipeline(candles, config))
        features_by_tf[tf] = features
        logger.info(f"[{symbol}] {tf}: {len(features)} velas con features validas ({csv_path.name})")

    feature_columns = select_feature_columns(features_by_tf["M15"])
    for tf in TIMEFRAMES:
        missing = set(feature_columns) - set(features_by_tf[tf].columns)
        if missing:
            raise RuntimeError(f"[{symbol}] Columnas de features inconsistentes en {tf}: faltan {missing}")

    seq_len_by_tf = config["model"]["sequence_length"]
    dataset = MultiTimeframeTradingDataset(
        features_by_tf, feature_columns, seq_len_by_tf,
        tp_atr_mult=config["model"]["tp_atr_mult"], sl_atr_mult=config["model"]["sl_atr_mult"],
    )
    logger.info(f"[{symbol}] Dataset: {len(dataset)} ejemplos alineados en las 4 temporalidades")
    return dataset, feature_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--data-dir", default=None, help="Por defecto, paths.raw_data de config.yaml")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para pesos iniciales y barajado")
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_settings()
    setup_logging(config["secrets"].log_level, config["paths"]["logs_dir"])

    data_dir = Path(args.data_dir or config["paths"]["raw_data"])

    reference_feature_columns: list[str] | None = None
    train_parts, val_parts, train_direction_parts = [], [], []

    for symbol in args.symbols:
        dataset, feature_columns = _build_dataset_for_symbol(symbol, data_dir, config)

        if reference_feature_columns is None:
            reference_feature_columns = feature_columns
        elif feature_columns != reference_feature_columns:
            raise RuntimeError(
                f"[{symbol}] feature_columns no coincide con los demas simbolos entrenados juntos."
            )

        split_idx = int(len(dataset) * (1 - args.val_split))
        train_parts.append(Subset(dataset, range(split_idx)))
        val_parts.append(Subset(dataset, range(split_idx, len(dataset))))
        train_direction_parts.append(dataset.direction[:split_idx])

    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    logger.info(
        f"Dataset combinado ({', '.join(args.symbols)}): "
        f"{len(train_dataset)} train / {len(val_dataset)} val"
    )

    # Ponderado por frecuencia inversa de clase (solo con la distribucion de train,
    # no de val) para que el modelo no colapse a predecir siempre la clase mayoritaria.
    class_counts = np.bincount(np.concatenate(train_direction_parts), minlength=3)
    class_weights = class_counts.sum() / (3 * np.maximum(class_counts, 1))
    logger.info(
        f"Distribucion direction (train) neutral/long/short: {class_counts.tolist()}, "
        f"pesos: {class_weights.round(3).tolist()}"
    )

    model = build_model(config, n_features=len(reference_feature_columns))
    trainer = Trainer(model, config, direction_class_weights=torch.tensor(class_weights, dtype=torch.float32))
    trainer.fit(train_dataset, val_dataset)

    out_name = f"{'-'.join(args.symbols)}_{config['model']['architecture']}.pt"
    out_path = Path(config["paths"]["models_dir"]) / out_name
    trainer.save_checkpoint(out_path, reference_feature_columns)


if __name__ == "__main__":
    main()
