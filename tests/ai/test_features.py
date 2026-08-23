from config.settings import load_yaml_config, CONFIG_YAML_PATH
from tradingai.ai.data.features.pipeline import build_feature_pipeline


def test_feature_pipeline_runs(synthetic_candles):
    config = load_yaml_config(CONFIG_YAML_PATH)
    features = build_feature_pipeline(synthetic_candles, config)

    assert len(features) == len(synthetic_candles)
    for col in ["swing_high", "swing_low", "bullish_ob", "bearish_ob", "bullish_fvg", "rsi_14", "atr_14"]:
        assert col in features.columns
