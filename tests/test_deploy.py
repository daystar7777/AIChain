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
