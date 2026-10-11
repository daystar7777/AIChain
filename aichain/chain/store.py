"""Persistent chain log (first slice of M0): an append-only list of transactions plus
periodic state-root checkpoints, in SQLite.

On start the board replays every transaction in order, advancing blocks to each
transaction's height in between; the deterministic rules rebuild exactly the same
state, which the checkpoints verify. Content bodies and each node's sealed (not yet
revealed) answers are stored too, so a restart loses nothing.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from aichain.board import Board
from aichain.chain.errors import ChainHalted

CHECKPOINT_EVERY = 120  # blocks (10 minutes at 5 s/block)


class ReplayMismatch(Exception):
    pass


class ChainStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS log (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, height INTEGER NOT NULL,
        type TEXT, body TEXT, state_root TEXT, ts REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS content (hash TEXT PRIMARY KEY, author TEXT, body TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tip (id INTEGER PRIMARY KEY CHECK (id = 1), height INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS sealed (
        answer_id TEXT PRIMARY KEY, node TEXT NOT NULL, body TEXT NOT NULL, salt TEXT NOT NULL);
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)
        self._lock = threading.Lock()

    def _write(self, sql: str, args: tuple) -> None:
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    # ---- log ----

    def append_tx(self, height: int, tx_type: str, body: dict[str, Any]) -> None:
        self._write("INSERT INTO log (kind, height, type, body, ts) VALUES ('tx', ?, ?, ?, ?)",
                    (height, tx_type, json.dumps(body, ensure_ascii=False), time.time()))

    def append_checkpoint(self, height: int, state_root: str) -> None:
        self._write("INSERT INTO log (kind, height, state_root, ts) VALUES ('checkpoint', ?, ?, ?)",
                    (height, state_root, time.time()))

    def entries(self) -> list[tuple[str, int, str | None, dict[str, Any] | None, str | None]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, height, type, body, state_root FROM log ORDER BY seq").fetchall()
        return [(k, h, t, json.loads(b) if b else None, r) for k, h, t, b, r in rows]

    def set_tip(self, height: int) -> None:
        self._write("INSERT INTO tip VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET height = excluded.height",
                    (height,))

    def tip(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT height FROM tip WHERE id = 1").fetchone()
        return row[0] if row else 0

    def tx_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM log WHERE kind = 'tx'").fetchone()[0]

    # ---- content and sealed answers ----

    def put_content(self, h: str, obj: Any, author: str | None) -> None:
        self._write("INSERT OR IGNORE INTO content VALUES (?, ?, ?)",
                    (h, author, json.dumps(obj, ensure_ascii=False)))

    def all_content(self) -> list[tuple[str, str | None, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT hash, author, body FROM content").fetchall()
        return [(h, a, json.loads(b)) for h, a, b in rows]

    def put_sealed(self, node: str, answer_id: str, body: dict[str, Any], salt: str) -> None:
        self._write("INSERT OR REPLACE INTO sealed VALUES (?, ?, ?, ?)",
                    (answer_id, node, json.dumps(body, ensure_ascii=False), salt))

    def delete_sealed(self, answer_id: str) -> None:
        self._write("DELETE FROM sealed WHERE answer_id = ?", (answer_id,))

    def sealed_for(self, node: str) -> dict[str, tuple[dict[str, Any], str]]:
        with self._lock:
            rows = self._conn.execute("SELECT answer_id, body, salt FROM sealed WHERE node = ?",
                                      (node,)).fetchall()
        return {a: (json.loads(b), s) for a, b, s in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def replay(board: Board, store: ChainStore) -> dict[str, int]:
    """Rebuild `board` from the log. Raises ReplayMismatch if a checkpoint disagrees."""
    txs = checks = 0
    for kind, height, tx_type, body, root in store.entries():
        if height > board.height:
            board.advance(height - board.height)  # events are discarded: nothing to wake on replay
        if kind == "tx":
            board.apply(tx_type, body)
            txs += 1
        else:
            if board.height == height and board.state_root() != root:
                raise ReplayMismatch(f"state root differs at height {height}")
            checks += 1
    if store.tip() > board.height:  # blocks produced after the last transaction
        board.advance(store.tip() - board.height)
    return {"transactions": txs, "checkpoints": checks, "height": board.height}


def attach(board: Board, store: ChainStore) -> dict[str, int]:
    """Replay, then record every new transaction and a checkpoint every CHECKPOINT_EVERY blocks."""
    stats = replay(board, store)

    def record(tx_type: str, body: dict) -> None:
        try:
            store.append_tx(board.height, tx_type, body)
        except Exception as e:  # disk full, locked or corrupt DB: fail closed
            board.halted = f"chain log write failed: {type(e).__name__}: {e}"
            raise ChainHalted(board.halted) from e

    board.recorder = record
    original = board._on_block

    def on_block():
        events = original()
        try:
            store.set_tip(board.height)
            if board.height % CHECKPOINT_EVERY == 0:
                store.append_checkpoint(board.height, board.state_root())
        except Exception as e:
            board.halted = f"chain log write failed: {type(e).__name__}: {e}"
            raise ChainHalted(board.halted) from e
        return events

    board._on_block = on_block
    return stats
