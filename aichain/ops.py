"""Operational helpers for running 24/7: a trace file that rotates and expires, and the
systemd notify protocol (readiness and watchdog) without extra dependencies.

The network writes its trace through `DailyTrace`: one JSONL file per UTC day, files older than
`keep_days` deleted. A failed write (disk full) is raised to the caller, which counts it and
goes on (`Network.log`); the file is reopened on the next write, so the trace resumes by itself
once space is freed.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TextIO


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of an env file (the format aichain.sim.factory.load_env reads), returned
    without touching os.environ, so secrets in it stay out of child processes."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def prune_files(directory: Path, pattern: str, keep_days: float, now: float | None = None) -> list[Path]:
    """Delete files matching `pattern` in `directory` last modified more than keep_days ago."""
    cutoff = (now if now is not None else time.time()) - keep_days * 86400
    removed = []
    for path in directory.glob(pattern):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed.append(path)
        except OSError:  # vanished or not ours: try again next time
            continue
    return removed


class DailyTrace:
    """File-like sink (`write`) for the JSONL trace: <directory>/<prefix>-YYYYMMDD.jsonl (UTC),
    appended across restarts, rotated at midnight, kept for keep_days."""

    def __init__(self, directory: Path, prefix: str = "bridge", keep_days: float = 14.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.directory = directory
        self.prefix = prefix
        self.keep_days = keep_days
        self._clock = clock
        self._day: str | None = None
        self._file: TextIO | None = None
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.directory / f"{self.prefix}-{self._today()}.jsonl"

    def _today(self) -> str:
        return time.strftime("%Y%m%d", time.gmtime(self._clock()))

    def write(self, text: str) -> int:
        day = self._today()
        if self._file is None or day != self._day:
            self.close()
            if day != self._day:  # a new day (or the first write): expire old files
                prune_files(self.directory, f"{self.prefix}-*.jsonl", self.keep_days, self._clock())
            self._day = day
            self._file = (self.directory / f"{self.prefix}-{day}.jsonl").open("a", encoding="utf-8", buffering=1)
        try:
            return self._file.write(text)
        except OSError:
            self.close()  # reopen on the next write
            raise

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def __enter__(self) -> "DailyTrace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---- systemd: Type=notify readiness and WatchdogSec= ----

def sd_notify(message: str) -> bool:
    """Send one message (e.g. "READY=1", "WATCHDOG=1") to $NOTIFY_SOCKET. False when the
    process does not run under systemd or the socket cannot be reached."""
    addr = os.environ.get("NOTIFY_SOCKET", "")
    if not addr:
        return False
    if addr.startswith("@"):  # abstract namespace
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(message.encode())
        return True
    except OSError:
        return False


class Watchdog:
    """Tells systemd the service is ready once the network loop has run, then pings the
    watchdog (WatchdogSec=) from its own thread as long as the loop keeps making progress.

    `last_progress()` returns the time.monotonic() of the last sign of life (a loop turn, a log
    entry, an LLM call). Episodes may block the loop for minutes on slow LLM calls, so the
    stall limit is generous; past it the pings stop and systemd kills and restarts the service.
    A hung process (no thread runs) stops pinging too. Without WATCHDOG_USEC it only sends
    READY=1 and STATUS= lines (both no-ops outside systemd)."""

    def __init__(self, last_progress: Callable[[], float], stall_seconds: float = 1200.0,
                 status: Callable[[], str] | None = None, interval: float | None = None) -> None:
        self.last_progress = last_progress
        self.stall_seconds = stall_seconds
        self.status = status
        usec = os.environ.get("WATCHDOG_USEC", "")
        pid = os.environ.get("WATCHDOG_PID", "")
        self.enabled = usec.isdigit() and (not pid.isdigit() or int(pid) == os.getpid())
        # systemd recommends pinging at half the timeout or more often.
        self.interval = interval or (int(usec) / 1e6 / 3 if self.enabled else 30.0)
        self.ready = False
        self.stalled = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="aichain-watchdog", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
            self._thread = None

    def _run(self) -> None:
        last_ping = 0.0
        while not self._stop.wait(min(1.0, self.interval)):
            now = time.monotonic()
            try:
                if (not self.ready or now - last_ping >= self.interval) and self.beat(now):
                    last_ping = now
            except Exception as e:  # e.g. the status line failed: skip this round, keep watching
                print(f"[aichain] watchdog: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def beat(self, now: float) -> bool:
        """READY=1 once the loop has run, then WATCHDOG=1 (and STATUS=) while it makes
        progress. Returns whether it pinged."""
        progress = self.last_progress()
        if not progress:
            return False  # the loop has not run yet (still starting)
        if not self.ready:
            self.ready = True
            sd_notify("READY=1")
        if now - progress >= self.stall_seconds:
            if not self.stalled:
                self.stalled = True
                print(f"[aichain] network loop made no progress for {now - progress:.0f} s; no more "
                      "watchdog pings (systemd will restart the service)", file=sys.stderr, flush=True)
            return False
        self.stalled = False
        lines = (["WATCHDOG=1"] if self.enabled else []) + ([f"STATUS={self.status()}"] if self.status else [])
        if lines:
            sd_notify("\n".join(lines))
        return True
