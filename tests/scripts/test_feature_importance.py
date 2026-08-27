import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

spec = importlib.util.spec_from_file_location("feature_importance", PROJECT_ROOT / "scripts" / "feature_importance.py")
feature_importance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_importance)


def test_identifies_structure_features_by_keyword():
    for name in [
        "D1_bullish_ob", "H1_liquidity_sweep_bearish", "M15_pd_zone", "M5_wyckoff_spring",
        "D1_elliott_impulse_confidence", "H1_regime_strong_trend", "M15_bias_bullish",
        "M5_session_london", "D1_is_killzone", "H1_dist_to_poc_pct", "M15_inside_value_area",
        "M5_vwap_zscore", "D1_bullish_momentum_divergence",
        "H1_swing_low", "M15_swing_high", "D1_asian_range_width_pct", "M15_broke_ny_or_range_low",
        "M15_is_accum_dist_zone", "M15_dist_to_va_low_pct",
    ]:
        assert feature_importance._is_structure_feature(name), f"{name} deberia ser estructura"


def test_does_not_flag_generic_indicators_as_structure():
    for name in ["D1_rsi_14", "H1_atr_pct", "M15_ema_20_pct", "M5_MACD_12_26_9", "D1_close_ret"]:
        assert not feature_importance._is_structure_feature(name), f"{name} NO deberia ser estructura"


def test_accuracy_scoring_matches_fraction_correct():
    import numpy as np

    class _FakeEstimator:
        def predict(self, X):
            return np.array([1, 1, 0, 2])

    y = np.array([1, 0, 0, 2])
    assert feature_importance._accuracy_scoring(_FakeEstimator(), None, y) == 0.75
