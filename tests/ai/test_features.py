from config.settings import load_yaml_config, CONFIG_YAML_PATH
from tradingai.ai.data.features.pipeline import build_feature_pipeline


def test_feature_pipeline_runs(synthetic_candles):
    config = load_yaml_config(CONFIG_YAML_PATH)
    features = build_feature_pipeline(synthetic_candles, config)

    assert len(features) == len(synthetic_candles)
    for col in [
        "swing_high",
        "swing_low",
        "bullish_ob",
        "bearish_ob",
        "bullish_fvg",
        "rsi_14",
        "atr_14",
        "liquidity_sweep_bullish",
        "liquidity_sweep_bearish",
        "bullish_fvg_inverted",
        "bearish_fvg_inverted",
        "session_asia",
        "session_london",
        "session_ny",
        "is_killzone",
        "vwap_dist_pct",
        "vwap_zscore",
        "wyckoff_spring",
        "wyckoff_upthrust",
        "elliott_impulse_confidence",
        "elliott_wave2_retrace_pct",
        "dist_to_poc_pct",
        "inside_value_area",
        "bearish_momentum_divergence",
        "bullish_momentum_divergence",
        "ADX_14",
    ]:
        assert col in features.columns
