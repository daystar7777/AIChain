"""Static checks of the systemd units in deploy/systemd (they are not run in tests)."""

import re
from pathlib import Path

UNITS = Path(__file__).resolve().parents[1] / "deploy" / "systemd"


def settings(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith(("#", "[")):
            key, _, value = line.partition("=")
            out.setdefault(key.strip(), []).append(value.strip())
    return out


def size(value: str) -> int:
    m = re.fullmatch(r"(\d+)([KMG]?)", value)
    assert m, value
    return int(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)]


def test_every_service_has_a_memory_ceiling():
    services = sorted(UNITS.glob("*.service"))
    assert len(services) >= 6
    for path in services:
        s = settings(path)
        assert "MemoryMax" in s and "MemoryHigh" in s, path.name
        assert size(s["MemoryHigh"][-1]) < size(s["MemoryMax"][-1]) <= 2 << 30, path.name


LONG_RUNNING = ["aichain-bridge.service", "aichain-browser.service", "aichain-egress.service",
                "aichain-journal.service", "aichain-telegram.service"]


def test_services_keep_retrying_and_alert_on_failure():
    for name in LONG_RUNNING:
        s = settings(UNITS / name)
        assert s["StartLimitIntervalSec"] == ["0"] and "StartLimitBurst" not in s, name
        assert s["OnFailure"] == ["aichain-alert@%n.service"], name
        assert s["Restart"] == ["on-failure"] and "RestartMaxDelaySec" in s, name
    for oneshot in ("aichain-report.service", "aichain-backup.service"):
        assert settings(UNITS / oneshot)["OnFailure"] == ["aichain-alert@%n.service"]
    alert = settings(UNITS / "aichain-alert@.service")
    assert "OnFailure" not in alert and "%i" in alert["ExecStart"][0]  # no alert loops
    assert "TELEGRAM" not in "".join(alert["ExecStart"])  # the token is read from the file, not argv


def test_bridge_unit_uses_notify_and_the_watchdog():
    s = settings(UNITS / "aichain-bridge.service")
    assert s["Type"] == ["notify"] and int(s["WatchdogSec"][0]) >= 60
    assert s["TimeoutStartSec"] == ["30min"]  # replaying a long chain log before READY=1


def test_backup_timer_runs_daily_with_the_public_key_only():
    timer = settings(UNITS / "aichain-backup.timer")
    assert timer["OnCalendar"] and timer["Persistent"] == ["true"]
    s = settings(UNITS / "aichain-backup.service")
    cmd = s["ExecStart"][0]
    assert "aichain.backup auto" in cmd and "--keep 7" in cmd and "/opt/aichain/backups" in cmd
    assert "PASSPHRASE" not in (UNITS / "aichain-backup.service").read_text()
