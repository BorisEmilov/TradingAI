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

    predictor = GBMPredictor(model, config, feature_columns)
    assert predictor.seq_len_by_tf == SEQ_LEN

    signal = predictor.predict(synthetic_multi_tf_candles, symbol="EURUSD")
    assert signal.direction in (Direction.LONG, Direction.SHORT, Direction.NEUTRAL)
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.entry_price == synthetic_multi_tf_candles["M15"]["close"].iloc[-1]
    if signal.direction != Direction.NEUTRAL:
        assert signal.take_profit is not None
        assert signal.stop_loss is not None
        assert signal.risk_reward_ratio == pytest.approx(2.0)

    # Checkpoint round-trip
    import joblib

    ckpt_path = tmp_path / "test_gbm.joblib"
    joblib.dump({"model": model, "feature_columns": feature_columns, "config": config}, ckpt_path)
    reloaded = GBMPredictor.from_checkpoint(ckpt_path)
    signal2 = reloaded.predict(synthetic_multi_tf_candles, symbol="EURUSD")
    assert signal2.direction == signal.direction


def test_gbm_predictor_raises_on_missing_timeframe(synthetic_multi_tf_candles):
    config = load_yaml_config(CONFIG_YAML_PATH)
    config = {**config, "model": {**config["model"], "sequence_length": SEQ_LEN}}
    model = HistGradientBoostingClassifier()
    predictor = GBMPredictor(model, config, feature_columns=["ema_20_pct"])

    incomplete = dict(synthetic_multi_tf_candles)
    del incomplete["M5"]
    try:
        predictor.predict(incomplete, symbol="EURUSD")
        assert False, "deberia haber lanzado ValueError"
    except ValueError:
        pass
