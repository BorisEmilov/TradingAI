import importlib.util
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

spec = importlib.util.spec_from_file_location("daily_review", PROJECT_ROOT / "scripts" / "daily_review.py")
daily_review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily_review)


def _write_trades_csv(path: Path, rows: list[str]) -> None:
    header = "timestamp,event,ticket,symbol,direction,price,sl,tp,confidence,lot_size,profit"
    path.write_text("\n".join([header, *rows]) + "\n")


def test_no_closed_trades_returns_empty_summary(tmp_path):
    path = tmp_path / "trades.csv"
    _write_trades_csv(path, ["2026-08-24T08:00:00+00:00,APERTURA,1,EURUSD,short,1.1000,1.1010,1.0980,0.80,3.0,"])

    summary = daily_review.run_review(date(2026, 8, 24), path, tmp_path / "tight_stop_trades.csv")

    assert summary["n_closed"] == 0


def test_computes_win_rate_and_pnl_correctly(tmp_path):
    path = tmp_path / "trades.csv"
    _write_trades_csv(
        path,
        [
            "2026-08-24T08:00:00+00:00,APERTURA,1,EURUSD,short,1.1000,1.1010,1.0980,0.80,3.0,",
            "2026-08-24T09:00:00+00:00,CIERRE,1,EURUSD,,1.0982,,,,,45.50",
            "2026-08-24T10:00:00+00:00,APERTURA,2,GBPJPY,long,190.00,189.50,191.00,0.90,2.0,",
            "2026-08-24T11:00:00+00:00,CIERRE,2,GBPJPY,,189.52,,,,,-60.20",
        ],
    )

    summary = daily_review.run_review(date(2026, 8, 24), path, tmp_path / "tight_stop_trades.csv")

    assert summary["n_closed"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["total_pnl"] == 45.50 - 60.20


def test_ignores_closes_from_other_dates(tmp_path):
    path = tmp_path / "trades.csv"
    _write_trades_csv(
        path,
        [
            "2026-08-23T08:00:00+00:00,APERTURA,1,EURUSD,short,1.1000,1.1010,1.0980,0.80,3.0,",
            "2026-08-23T09:00:00+00:00,CIERRE,1,EURUSD,,1.0982,,,,,45.50",
        ],
    )

    summary = daily_review.run_review(date(2026, 8, 24), path, tmp_path / "tight_stop_trades.csv")

    assert summary["n_closed"] == 0


def test_confidence_bucket_rounds_down_to_5_percent():
    assert daily_review._confidence_bucket(0.77) == "75-80%"
    assert daily_review._confidence_bucket(0.90) == "90-95%"


def test_flags_trade_with_tight_stop_loss(tmp_path):
    # GBPUSD con SL a solo 0.0007 (7 pips) -- el caso real del 2026-08-25 que motivo
    # este mecanismo de deteccion.
    path = tmp_path / "trades.csv"
    tight_stop_path = tmp_path / "tight_stop_trades.csv"
    _write_trades_csv(
        path,
        ["2026-08-25T07:00:04+00:00,APERTURA,58140748366,GBPUSD,short,1.36286,1.36356,1.36150,0.84,2.72,"],
    )

    daily_review.run_review(date(2026, 8, 25), path, tight_stop_path)

    assert tight_stop_path.exists()
    flagged = tight_stop_path.read_text()
    assert "58140748366" in flagged


def test_does_not_flag_trade_with_normal_stop_loss(tmp_path):
    path = tmp_path / "trades.csv"
    tight_stop_path = tmp_path / "tight_stop_trades.csv"
    _write_trades_csv(
        path,
        ["2026-08-25T07:00:04+00:00,APERTURA,1,GBPUSD,short,1.36286,1.36486,1.35886,0.84,2.72,"],
    )

    daily_review.run_review(date(2026, 8, 25), path, tight_stop_path)

    assert not tight_stop_path.exists()


def test_does_not_flag_the_same_ticket_twice_across_reruns(tmp_path):
    path = tmp_path / "trades.csv"
    tight_stop_path = tmp_path / "tight_stop_trades.csv"
    _write_trades_csv(
        path,
        ["2026-08-25T07:00:04+00:00,APERTURA,58140748366,GBPUSD,short,1.36286,1.36356,1.36150,0.84,2.72,"],
    )

    daily_review.run_review(date(2026, 8, 25), path, tight_stop_path)
    daily_review.run_review(date(2026, 8, 25), path, tight_stop_path)

    assert tight_stop_path.read_text().count("58140748366") == 1
