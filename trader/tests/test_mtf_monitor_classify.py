"""Watchdog (scripts/mtf_monitor.sh): distinguir parada pedida de caída por error
a partir del estado/código de salida que reporta systemd."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mtf_monitor.sh"


def _classify(active, sub, status):
    out = subprocess.run(["bash", "-c", f'MTF_MONITOR_LIB=1 source "{SCRIPT}"; classify_pilot "$@"', "_", active, sub, status],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.mark.parametrize("active,sub,status,expected", [
    ("active", "running", "0", "running"),
    ("inactive", "dead", "0", "stopped"),            # stop flag / systemctl stop -> no es caída
    ("inactive", "dead", "1", "crashed"),            # pilot_crashed
    ("activating", "auto-restart", "1", "crashed"),  # caído, systemd esperando para relanzar
    ("activating", "auto-restart", "9", "crashed"),  # SIGKILL
    ("inactive", "dead", "3", "already_running"),
    ("failed", "failed", "1", "crashloop"),
])
def test_watchdog_distinguishes_requested_stop_from_crash(active, sub, status, expected):
    assert _classify(active, sub, status) == expected


def test_watchdog_logs_termination_signal_before_exiting():
    import signal
    import time
    proc = subprocess.Popen(["bash", str(SCRIPT)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.5)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert "señal de terminación recibida" in out and "line" not in out  # sin errores de sintaxis del trap
