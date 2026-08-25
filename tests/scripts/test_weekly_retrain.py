import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

spec = importlib.util.spec_from_file_location("weekly_retrain", PROJECT_ROOT / "scripts" / "weekly_retrain.py")
weekly_retrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weekly_retrain)


def test_promotes_when_majority_folds_positive():
    folds = [{"win_rate": 50.0, "pnl": 5.0}, {"win_rate": 48.0, "pnl": 3.0}, {"win_rate": 40.0, "pnl": -1.0}, {"win_rate": 55.0, "pnl": 4.0}]
    assert weekly_retrain._should_promote(folds)


def test_does_not_promote_when_avg_pnl_negative():
    folds = [{"win_rate": 30.0, "pnl": -5.0}, {"win_rate": 28.0, "pnl": -3.0}, {"win_rate": 60.0, "pnl": 4.0}]
    assert not weekly_retrain._should_promote(folds)


def test_does_not_promote_when_majority_folds_negative():
    # 1 de 4 folds positivos, aunque el pnl medio de casualidad diera positivo.
    folds = [{"win_rate": 20.0, "pnl": -1.0}, {"win_rate": 25.0, "pnl": -1.0}, {"win_rate": 30.0, "pnl": -1.0}, {"win_rate": 70.0, "pnl": 10.0}]
    assert not weekly_retrain._should_promote(folds)


def test_does_not_promote_with_no_folds():
    assert not weekly_retrain._should_promote([])


def test_fold_line_regex_parses_real_log_format():
    sample = "Fold 2 BACKTEST (comparable a walk_forward.py): 121 operaciones, win_rate=44.6%, pnl=-0.83%"
    matches = weekly_retrain._FOLD_LINE.findall(sample)
    assert matches == [("44.6", "-0.83")]


def test_risk_line_regex_parses_real_log_format():
    sample = "Fold 2 riesgo: sharpe=0.323, sortino=0.600, max_drawdown=2.00%"
    matches = weekly_retrain._RISK_LINE.findall(sample)
    assert matches == [("0.323", "0.600", "2.00")]
