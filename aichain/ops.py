"""Operational helpers for running 24/7: a trace file that rotates and expires.

The network writes its trace through `DailyTrace`: one JSONL file per UTC day, files older than
`keep_days` deleted. A failed write (disk full) is raised to the caller, which counts it and
goes on (`Network.log`); the file is reopened on the next write, so the trace resumes by itself
once space is freed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, TextIO


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
