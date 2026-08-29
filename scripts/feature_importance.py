"""Importancia por permutacion de las features del GBM en produccion.

Pregunta que responde con evidencia (no con impresion visual mirando graficos, ver
conversacion 2026-08-26): ¿el modelo realmente usa las features de estructura (order
blocks, liquidity sweeps, premium/discount, Wyckoff, Elliott, regimen, sesgo
D1/H1/M15/M5, sesiones/killzones) o se apoya sobre todo en otra cosa (momentum/RSI
crudo) y las ignora de facto?

`HistGradientBoostingClassifier` no expone `feature_importances_` (a diferencia de
RandomForest) -- se mide con importancia por permutacion de sklearn sobre el ultimo
fold del walk-forward (fuera de muestra), barajando cada columna y viendo cuanto
empeora la accuracy.

Uso:
    python scripts/feature_importance.py --symbol EURUSD
    python scripts/feature_importance.py --symbol GBPJPY --top-n 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import threadpoolctl  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from config.settings import get_settings  # noqa: E402
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns  # noqa: E402
from tradingai.ai.data.loader import load_csv  # noqa: E402
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, flatten_last_timestep  # noqa: E402
from tradingai.ai.data.preprocessor import normalize_ohlcv  # noqa: E402
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset  # noqa: E402
from tradingai.utils.thermal import wait_for_safe_temp  # noqa: E402

from loguru import logger  # noqa: E402

# Palabras clave que identifican una feature como "de estructura" para el reporte --
# el resto (ema/rsi/atr/macd/bollinger crudos, retornos) se trata como "generica".
# NOTA (2026-08-26): la primera version de esta lista se quedaba corta -- no
# reconocia swing_high/low (estructura de mercado), asian_range/or_range (rango de
# sesion), accum_dist (Wyckoff/flujo de volumen) ni dist_to_va (value area), asi que
# subestimaba la proporcion real de "estructura" en el primer corrido real (EURUSD).
_STRUCTURE_KEYWORDS = [
    "ob", "fvg", "sweep", "liquidity", "pd_zone", "wyckoff", "elliott",
    "regime", "bias", "session", "killzone", "poc", "value_area", "vwap", "divergence",
    "swing", "asian_range", "or_range", "accum_dist", "dist_to_va",
]


def _is_structure_feature(name: str) -> bool:
    return any(kw in name.lower() for kw in _STRUCTURE_KEYWORDS)


class _EnsembleWrapper:
    """Envoltorio minimo para que `permutation_importance` trate el ensemble de N
    seeds como un solo estimador -- promedia predict_proba, igual que hace
    `GBMPredictor.predict()` en produccion.

    `fit()` es un no-op: sklearn exige que el objeto "parezca" un estimador
    entrenable (valida `hasattr(estimator, "fit")`), pero `permutation_importance`
    nunca lo llama de verdad -- los modelos ya vienen entrenados desde afuera.
    """

    def __init__(self, models: list) -> None:
        self.models = models
        self.classes_ = models[0].classes_

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Sin esto, cada predict_proba abre su propio pool de hilos BLAS/OpenMP --
        # permutation_importance llama a esto cientos de veces (n_repeats x n_features),
        # y sin limitar ya se vio en vivo que eso satura la CPU (ver
        # gbm_predictor.py y memoria feedback-hardware-thermal-limits).
        with threadpoolctl.threadpool_limits(limits=1):
            probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return self.classes_[np.argmax(probs, axis=1)]


def _make_accuracy_scoring(max_temp_c: float):
    # permutation_importance llama al scoring cientos de veces (n_repeats x
    # n_features) en un solo bloque sin retornar el control -- sin este chequeo
    # aca, el unico gate termico era el de entrenar el ensemble, y esta fase (mas
    # larga) corria completamente sin proteccion. Bug real encontrado en vivo el
    # 2026-08-29: 94-95C sostenido, mismo patron que el gap de train_gbm.py.
    def _accuracy_scoring(estimator, X, y) -> float:
        wait_for_safe_temp(max_temp_c=max_temp_c)
        return float((estimator.predict(X) == y).mean())

    return _accuracy_scoring


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--n-folds", type=int, default=5, help="Mismos limites que baseline_gbm.py; solo se usa el ULTIMO fold como test")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-seeds", type=int, default=5, help="Igual que train_gbm.py -- mismo ensemble que produccion")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument("--max-temp-c", type=float, default=80.0)
    args = parser.parse_args()

    config = get_settings()
    data_dir = Path(args.data_dir or config["paths"]["raw_data"])

    # IMPORTANTE: no se reutiliza el checkpoint de produccion (data/models/*_gbm.joblib)
    # -- ese se entrena sobre el dataset COMPLETO (train_gbm.py, sin train/test split,
    # a proposito: para produccion se quiere usar todo el historico disponible). Medir
    # importancia contra el sino implicaria evaluar sobre datos que el modelo ya vio
    # entrenando -- bug real encontrado el 2026-08-26: daba 99.1% de accuracy "en test",
    # numero sin sentido para prediccion de direccion FX. Aca se entrena un modelo
    # PROPIO solo para esta medicion, con el mismo split purgado que baseline_gbm.py
    # (fuera de muestra de verdad), usando el esquema de features ACTUAL (incluye
    # features nuevas como bias_* aunque el checkpoint de produccion no las tenga
    # todavia).
    candles_by_tf = {tf: load_csv(data_dir / f"{args.symbol}_{tf}.csv") for tf in TIMEFRAMES}
    features_by_tf = {tf: normalize_ohlcv(build_feature_pipeline(candles_by_tf[tf], config)) for tf in TIMEFRAMES}
    feature_columns = select_feature_columns(features_by_tf["M15"])

    dataset = MultiTimeframeTradingDataset(
        features_by_tf, feature_columns, config["model"]["sequence_length"],
        tp_atr_mult=config["model"]["tp_atr_mult"], sl_atr_mult=config["model"]["sl_atr_mult"],
    )
    X = flatten_last_timestep(dataset.sequences)
    y = dataset.direction
    n = len(dataset)

    fold_bounds = [int(n * i / args.n_folds) for i in range(args.n_folds + 1)]
    test_start, test_end = fold_bounds[-2], fold_bounds[-1]
    # Purge (Lopez de Prado, igual que baseline_gbm.py/walk_forward.py): nunca
    # entrenar sobre los ultimos `horizon` ejemplos antes de test_start, porque su
    # etiqueta mira hacia adelante y podria fugar informacion de test.
    purged_train_end = test_start - dataset.horizon
    X_train, y_train = X[:purged_train_end], y[:purged_train_end]
    X_test, y_test = X[test_start:test_end], y[test_start:test_end]
    logger.info(f"[{args.symbol}] Train (purgado): {len(y_train)}  Test (fuera de muestra): {len(y_test)}")

    models = []
    with threadpoolctl.threadpool_limits(limits=args.max_threads):
        for i in range(args.n_seeds):
            wait_for_safe_temp(max_temp_c=args.max_temp_c)
            m = HistGradientBoostingClassifier(random_state=42 + i, max_iter=200, early_stopping=True)
            m.fit(X_train, y_train)
            models.append(m)
    ensemble = _EnsembleWrapper(models)
    scoring = _make_accuracy_scoring(args.max_temp_c)

    baseline_acc = scoring(ensemble, X_test, y_test)
    logger.info(f"[{args.symbol}] Accuracy fuera de muestra: {baseline_acc * 100:.1f}%")

    result = permutation_importance(
        ensemble, X_test, y_test, n_repeats=args.n_repeats, random_state=42, scoring=scoring,
    )

    column_names = [f"{tf}_{col}" for tf in TIMEFRAMES for col in feature_columns]
    ranked = sorted(zip(column_names, result.importances_mean), key=lambda item: -item[1])

    print(f"\n=== Importancia por permutacion -- {args.symbol} (accuracy base {baseline_acc * 100:.1f}%) ===\n")
    print(f"{'#':>3} {'Feature':45s} {'Importancia':>12s}  Tipo")
    for i, (name, imp) in enumerate(ranked[: args.top_n], start=1):
        tag = "ESTRUCTURA" if _is_structure_feature(name) else ""
        print(f"{i:3d} {name:45s} {imp:12.5f}  {tag}")

    structure_sum = sum(imp for name, imp in ranked if _is_structure_feature(name))
    other_sum = sum(imp for name, imp in ranked if not _is_structure_feature(name))
    print(f"\nSuma de importancia -- features de ESTRUCTURA: {structure_sum:.4f}")
    print(f"Suma de importancia -- resto (momentum/precio/volumen crudo): {other_sum:.4f}")
    if structure_sum + other_sum > 0:
        pct = structure_sum / (structure_sum + other_sum) * 100
        print(f"Proporcion que corresponde a estructura: {pct:.1f}%")


if __name__ == "__main__":
    main()
