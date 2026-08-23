from config.settings import CONFIG_YAML_PATH, load_yaml_config
from tradingai.ai.data.features.pipeline import build_feature_pipeline, select_feature_columns
from tradingai.ai.data.multi_timeframe import TIMEFRAMES, align_and_build_sequences, last_closed_bar_index
from tradingai.ai.data.preprocessor import normalize_ohlcv
from tradingai.ai.training.dataset import MultiTimeframeTradingDataset

SEQ_LEN = {"D1": 5, "H1": 5, "M15": 5, "M5": 5}


def _build_features_by_tf(synthetic_multi_tf_candles):
    config = load_yaml_config(CONFIG_YAML_PATH)
    features_by_tf = {
        tf: normalize_ohlcv(build_feature_pipeline(df, config)) for tf, df in synthetic_multi_tf_candles.items()
    }
    feature_columns = select_feature_columns(features_by_tf["M15"])
    return features_by_tf, feature_columns


def test_align_and_build_sequences_shapes(synthetic_multi_tf_candles):
    features_by_tf, feature_columns = _build_features_by_tf(synthetic_multi_tf_candles)

    aligned = align_and_build_sequences(features_by_tf, feature_columns, SEQ_LEN)

    assert len(aligned.anchor_positions) > 0
    for tf in TIMEFRAMES:
        assert aligned.sequences[tf].shape == (
            len(aligned.anchor_positions),
            SEQ_LEN[tf],
            len(feature_columns),
        )


def test_no_lookahead_in_alignment(synthetic_multi_tf_candles):
    """La ultima barra usada en cada ventana no puede ser posterior al timestamp del anchor M15."""
    features_by_tf, feature_columns = _build_features_by_tf(synthetic_multi_tf_candles)
    aligned = align_and_build_sequences(features_by_tf, feature_columns, SEQ_LEN)

    m15_ts = features_by_tf["M15"]["timestamp"].to_numpy()
    anchor_ts = m15_ts[aligned.anchor_positions]

    for tf in TIMEFRAMES:
        tf_ts = features_by_tf[tf]["timestamp"].to_numpy()
        last_bar_idx = last_closed_bar_index(tf_ts, anchor_ts)
        assert (last_bar_idx >= 0).all()
        assert (tf_ts[last_bar_idx] <= anchor_ts).all()


def test_multi_timeframe_dataset(synthetic_multi_tf_candles):
    features_by_tf, feature_columns = _build_features_by_tf(synthetic_multi_tf_candles)

    dataset = MultiTimeframeTradingDataset(features_by_tf, feature_columns, SEQ_LEN, horizon=10)
    assert len(dataset) > 0

    item = dataset[0]
    for tf in TIMEFRAMES:
        assert item[tf].shape == (SEQ_LEN[tf], len(feature_columns))
    assert item["direction"].item() in (0, 1, 2)
    assert item["entry_timeframe"].item() in (0, 1)
