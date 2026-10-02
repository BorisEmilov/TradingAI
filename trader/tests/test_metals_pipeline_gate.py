"""El pipeline de metales debe quedar bloqueado hasta n>=50 sin abandono + luz verde del usuario."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import metals_pipeline as mp  # noqa: E402


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(mp, "GREENLIGHT", tmp_path / "greenlight.txt")
    return tmp_path


def _status(paths, n, abandono=()):
    (paths / "status.json").write_text(json.dumps({"criterio": {"n": n, "objetivo_n": 50, "abandono": list(abandono)}}))


def test_blocked_without_status(paths):
    assert mp.gate_status()[0] is False


def test_blocked_below_50(paths):
    _status(paths, 49)
    (paths / "greenlight.txt").write_text("ok")
    assert mp.gate_status() == (False, "bloqueado: piloto en n=49/50")


def test_cancelled_if_abandonment_fired(paths):
    _status(paths, 60, ["R medio -0.4 < -0.3 con n=20"])
    (paths / "greenlight.txt").write_text("ok")
    ok, why = mp.gate_status()
    assert not ok and why.startswith("CANCELADO")


def test_needs_user_greenlight_even_at_50(paths):
    _status(paths, 50)
    assert mp.gate_status()[0] is False
    (paths / "greenlight.txt").write_text("ok")
    assert mp.gate_status()[0] is True
