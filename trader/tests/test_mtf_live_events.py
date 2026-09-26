from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from trader.mtf_strategies.live_events import (
    format_human_line,
    log_event,
    notify_desktop,
    write_human_event,
    write_json_event,
)


def test_format_human_line_is_tail_f_friendly_single_line():
    line = format_human_line("2024-01-01T09:00:00+00:00", "EURUSD", "continuation", "position_opened", "entry=1.10 sl=1.09")
    assert "\n" not in line
    assert "EURUSD" in line and "continuation" in line and "position_opened" in line and "entry=1.10" in line


def test_write_json_event_appends_one_json_line(tmp_path):
    path = tmp_path / "events.jsonl"
    write_json_event(path, "position_opened", symbol="EURUSD", entry=1.1)
    write_json_event(path, "position_closed", symbol="EURUSD", exit=1.11)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["kind"] == "position_opened"
    assert row0["symbol"] == "EURUSD"
    assert row0["entry"] == 1.1
    assert "ts" in row0


def test_write_human_event_appends_one_readable_line(tmp_path):
    path = tmp_path / "events.txt"
    write_human_event(path, "EURUSD", "continuation", "position_opened", "entry=1.10")
    write_human_event(path, "GBPUSD", "reversal", "position_closed", "exit=1.25 r=1.0")

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert "EURUSD" in lines[0] and "position_opened" in lines[0]
    assert "GBPUSD" in lines[1] and "position_closed" in lines[1]


@patch("trader.mtf_strategies.live_events._NOTIFY_SEND_AVAILABLE", False)
def test_notify_desktop_returns_false_gracefully_when_unavailable():
    assert notify_desktop("title", "body") is False


@patch("trader.mtf_strategies.live_events._NOTIFY_SEND_AVAILABLE", True)
@patch("trader.mtf_strategies.live_events.subprocess.run")
def test_notify_desktop_invokes_notify_send_when_available(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

    ok = notify_desktop("Posición abierta", "EURUSD long @ 1.10")

    assert ok is True
    args = mock_run.call_args.args[0]
    assert args[0] == "notify-send"
    assert "Posición abierta" in args
    assert "EURUSD long @ 1.10" in args


@patch("trader.mtf_strategies.live_events._NOTIFY_SEND_AVAILABLE", True)
@patch("trader.mtf_strategies.live_events.subprocess.run", side_effect=OSError("boom"))
def test_notify_desktop_never_raises_even_if_subprocess_fails(mock_run):
    assert notify_desktop("title", "body") is False


def test_log_event_writes_all_channels_and_desktop_only_when_titled(tmp_path):
    json_path = tmp_path / "events.jsonl"
    human_path = tmp_path / "events.txt"

    with patch("trader.mtf_strategies.live_events.notify_desktop") as mock_notify:
        log_event(json_path, human_path, "EURUSD", "continuation", "position_opened", "entry=1.10",
                   desktop_title="Posición abierta")
        mock_notify.assert_called_once_with("Posición abierta", "entry=1.10")

    with patch("trader.mtf_strategies.live_events.notify_desktop") as mock_notify:
        log_event(json_path, human_path, "EURUSD", "continuation", "heartbeat", "still alive")
        mock_notify.assert_not_called()

    assert json_path.exists() and human_path.exists()
    assert len(json_path.read_text().splitlines()) == 2
    assert len(human_path.read_text().splitlines()) == 2
