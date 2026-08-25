import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from config.settings import CONFIG_YAML_PATH, load_yaml_config
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns
from tradingai.ai.data.multi_timeframe import flatten_last_timestep
from tradingai.ai.data.preprocessor import normalize_ohlcv
from tradingai.ai.inference.gbm_predictor import GBMPredictor
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset
from tradingai.core.signal import Direction

SEQ_LEN = {"D1": 5, "H1": 5, "M15": 5, "M5": 5}


def test_gbm_predictor_end_to_end(synthetic_multi_tf_candles, tmp_path):
    config = load_yaml_config(CONFIG_YAML_PATH)
    config = {**config, "model": {**config["model"], "sequence_length": SEQ_LEN}}

    features_by_tf = {
        tf: normalize_ohlcv(build_feature_pipeline(df, config)) for tf, df in synthetic_multi_tf_candles.items()
    }
    feature_columns = select_feature_columns(features_by_tf["M15"])
    dataset = MultiTimeframeTradingDataset(features_by_tf, feature_columns, SEQ_LEN)

    X = flatten_last_timestep(dataset.sequences)
    y = dataset.direction
    assert X.shape == (len(dataset), 4 * len(feature_columns))

    model = HistGradientBoostingClassifier(random_state=0, max_iter=20)
    model.fit(X, y)

    predictor = GBMPredictor([model], config, feature_columns)
    assert predictor.seq_len_by_tf == SEQ_LEN

    signal = predictor.predict(synthetic_multi_tf_candles, symbol="EURUSD")
    assert signal.direction in (Direction.LONG, Direction.SHORT, Direction.NEUTRAL)
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.entry_price == synthetic_multi_tf_candles["M15"]["close"].iloc[-1]
    if signal.direction != Direction.NEUTRAL:
        assert signal.take_profit is not None
        assert signal.stop_loss is not None
        assert signal.risk_reward_ratio == pytest.approx(2.0)

    # Checkpoint round-trip -- formato viejo, un solo modelo bajo "model" (singular),
    # de antes del ensemble. Debe seguir cargando bien (ensemble de tamano 1).
    import joblib

    ckpt_path = tmp_path / "test_gbm.joblib"
    joblib.dump({"model": model, "feature_columns": feature_columns, "config": config}, ckpt_path)
    reloaded = GBMPredictor.from_checkpoint(ckpt_path)
    assert len(reloaded.models) == 1
    signal2 = reloaded.predict(synthetic_multi_tf_candles, symbol="EURUSD")
    assert signal2.direction == signal.direction

    # Checkpoint round-trip -- formato nuevo, ensemble bajo "models" (plural).
    ckpt_path_ensemble = tmp_path / "test_gbm_ensemble.joblib"
    model2 = HistGradientBoostingClassifier(random_state=1, max_iter=20)
    model2.fit(X, y)
    joblib.dump({"models": [model, model2], "feature_columns": feature_columns, "config": config}, ckpt_path_ensemble)
    reloaded_ensemble = GBMPredictor.from_checkpoint(ckpt_path_ensemble)
    assert len(reloaded_ensemble.models) == 2
    reloaded_ensemble.predict(synthetic_multi_tf_candles, symbol="EURUSD")  # no debe reventar


def test_gbm_predictor_averages_probabilities_across_ensemble(synthetic_multi_tf_candles):
    import numpy as np

    config = load_yaml_config(CONFIG_YAML_PATH)
    config = {**config, "model": {**config["model"], "sequence_length": SEQ_LEN}}

    class _FixedProbaModel:
        def __init__(self, probs):
            self._probs = np.array([probs])

        def predict_proba(self, x):
            return self._probs

    # 2 modelos "opuestos" en su prediccion -- el promedio debe caer en el medio,
    # no coincidir con ninguno de los dos individualmente.
    model_a = _FixedProbaModel([0.1, 0.8, 0.1])  # confiado en LONG
    model_b = _FixedProbaModel([0.1, 0.2, 0.7])  # confiado en SHORT

    predictor = GBMPredictor([model_a, model_b], config, feature_columns=["ema_20_pct"])
    # Se sustituye build_feature_pipeline indirectamente: alcanza con un feature_columns
    # de 1 sola columna real para no depender de todo el pipeline en este test unitario.

    signal = predictor.predict(synthetic_multi_tf_candles, symbol="EURUSD")
    expected_probs = [(0.1 + 0.1) / 2, (0.8 + 0.2) / 2, (0.1 + 0.7) / 2]
    assert signal.rationale["direction_probs"] == pytest.approx(expected_probs)
    assert signal.direction == Direction.LONG  # promedio [neutral,long,short]=[0.1,0.5,0.4] -> gana long


def test_gbm_predictor_raises_on_missing_timeframe(synthetic_multi_tf_candles):
    config = load_yaml_config(CONFIG_YAML_PATH)
    config = {**config, "model": {**config["model"], "sequence_length": SEQ_LEN}}
    model = HistGradientBoostingClassifier()
    predictor = GBMPredictor([model], config, feature_columns=["ema_20_pct"])

    incomplete = dict(synthetic_multi_tf_candles)
    del incomplete["M5"]
    try:
        predictor.predict(incomplete, symbol="EURUSD")
        assert False, "deberia haber lanzado ValueError"
    except ValueError:
        pass
