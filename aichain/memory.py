"""Node memory and knowledge (design doc 04, ch. 4).

Append-only records with source and trust, hash-chained per node like the node chain
they stand in for. With a `KnowledgeStore` the records persist across runs in SQLite,
so what a node learns from the web stays with it. Recall is keyword overlap × trust ×
recency; superseded records are replaced by their latest correction.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aichain.common import canonical_json, sha256_hex

_WORD = re.compile(r"[\w가-힣]+")
GENESIS = "0" * 64


@dataclass
class MemoryRecord:
    seq: int
    height: int
    kind: str
    source: str
    trust: float
    content: str
    tags: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    supersedes: str | None = None
    topic: str | None = None
    data: dict[str, Any] | None = None  # structured body, e.g. a note's facts
    ts: float = 0.0
    prev_hash: str = GENESIS
    hash: str = ""

    def body(self) -> dict[str, Any]:
        return {"seq": self.seq, "height": self.height, "kind": self.kind, "source": self.source,
                "trust_e6": int(round(self.trust * 1_000_000)), "content": self.content,
                "tags": self.tags, "refs": self.refs, "supersedes": self.supersedes,
                "topic": self.topic, "data": self.data, "prev": self.prev_hash}


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 1}


class KnowledgeStore:
    """SQLite store for every node's memory records (one DB, one chain per node)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        node TEXT NOT NULL, seq INTEGER NOT NULL, hash TEXT NOT NULL, prev_hash TEXT NOT NULL,
        ts REAL NOT NULL, height INTEGER NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL,
        trust REAL NOT NULL, content TEXT NOT NULL, tags TEXT NOT NULL, refs TEXT NOT NULL,
        supersedes TEXT, topic TEXT, data TEXT,
        PRIMARY KEY (node, seq)
    );
    CREATE INDEX IF NOT EXISTS memories_topic ON memories(node, kind, topic);
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)
        self._lock = threading.Lock()

    def load(self, node: str) -> list[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, height, kind, source, trust, content, tags, refs, supersedes, topic, "
                "data, ts, prev_hash, hash FROM memories WHERE node = ? ORDER BY seq", (node,)).fetchall()
        return [MemoryRecord(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6]), json.loads(r[7]),
                             r[8], r[9], json.loads(r[10]) if r[10] else None, r[11], r[12], r[13])
                for r in rows]

    def append(self, node: str, rec: MemoryRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (node, rec.seq, rec.hash, rec.prev_hash, rec.ts, rec.height, rec.kind, rec.source,
                 rec.trust, rec.content, json.dumps(rec.tags, ensure_ascii=False),
                 json.dumps(rec.refs, ensure_ascii=False), rec.supersedes, rec.topic,
                 json.dumps(rec.data, ensure_ascii=False) if rec.data is not None else None))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Memory:
    def __init__(self, node_id: str = "", store: KnowledgeStore | None = None) -> None:
        self.node_id = node_id
        self.store = store
        self.records: list[MemoryRecord] = store.load(node_id) if store else []
        self._by_hash = {r.hash: r for r in self.records}

    def remember(self, height: int, kind: str, source: str, trust: float, content: str,
                 tags: list[str] | None = None, refs: list[str] | None = None, *,
                 supersedes: str | None = None, topic: str | None = None,
                 data: dict[str, Any] | None = None) -> MemoryRecord:
        prev = self.records[-1].hash if self.records else GENESIS
        rec = MemoryRecord(len(self.records) + 1, height, kind, source, trust, content,
                           list(tags or []), list(refs or []), supersedes, topic, data,
                           ts=time.time(), prev_hash=prev)
        rec.hash = sha256_hex(canonical_json(rec.body()))
        self.records.append(rec)
        self._by_hash[rec.hash] = rec
        if self.store:
            self.store.append(self.node_id, rec)
        return rec

    def verify_chain(self) -> bool:
        prev = GENESIS
        for r in self.records:
            if r.prev_hash != prev or sha256_hex(canonical_json(r.body())) != r.hash:
                return False
            prev = r.hash
        return True

    def superseded(self) -> set[str]:
        return {r.supersedes for r in self.records if r.supersedes}

    def latest_note(self, topic: str) -> MemoryRecord | None:
        dead = self.superseded()
        for r in reversed(self.records):
            if r.kind == "note" and r.topic == topic and r.hash not in dead:
                return r
        return None

    def notes(self) -> list[MemoryRecord]:
        dead = self.superseded()
        return [r for r in self.records if r.kind == "note" and r.hash not in dead]

    def recall(self, query: str, k: int = 5) -> list[MemoryRecord]:
        q = _tokens(query)
        if not q or not self.records:
            return []
        dead = self.superseded()
        newest = self.records[-1].seq
        scored = []
        for r in self.records:
            if r.hash in dead:
                continue
            words = _tokens(r.content) | {t.lower() for t in r.tags} | _tokens(r.topic or "")
            overlap = len(q & words)
            if overlap == 0:
                continue
            recency = 1 / (1 + (newest - r.seq) / 50)
            weight = 1.5 if r.kind == "note" else 1.0
            scored.append((overlap * r.trust * recency * weight, r.seq, r))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [r for _, _, r in scored[:k]]
