"""Node memory and knowledge (design doc 04, ch. 4).

Append-only records with source and trust, hash-chained per node like the node chain
they stand in for. With a `KnowledgeStore` the records persist across runs in SQLite,
so what a node learns from the web stays with it. Recall is keyword overlap × trust ×
recency; superseded records are replaced by their latest correction.

Storage quota: only what the Mastermind wrote (remember, note) and still holds — the latest
version of each note, nothing it forgot — counts. Core's own records (episode summaries,
settlements, deliveries) do not; instead Core compacts them: records older than a week are
replaced by a periodic digest. A compacted or forgotten record keeps its place in the hash
chain as a stub (hash and link, no content) named by the later digest or forget record.
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
MASTERMIND = "mastermind"  # MemoryRecord.by for records the model wrote through its tools
CORE_KEEP_BLOCKS = 7 * 17_280  # Core records stay whole for a week...
DIGEST_EVERY_BLOCKS = 17_280  # ...and are compacted into a digest once a day after that
DIGEST_MAX_CHARS = 2000
# Core records that stay whole: persona versions, reflection markers, digests and forget markers.
_KEEP_KINDS = {"persona", "reflection", "digest", "forget"}


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
    by: str | None = None  # MASTERMIND, or None for Core
    pruned: bool = False  # compacted or forgotten: only the hash and the link remain

    def body(self) -> dict[str, Any]:
        body = {"seq": self.seq, "height": self.height, "kind": self.kind, "source": self.source,
                "trust_e6": int(round(self.trust * 1_000_000)), "content": self.content,
                "tags": self.tags, "refs": self.refs, "supersedes": self.supersedes,
                "topic": self.topic, "data": self.data, "prev": self.prev_hash}
        if self.by is not None:  # absent from older records, whose hashes stay valid
            body["by"] = self.by
        return body


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
    # Columns added after the first schema (ALTER TABLE on older databases).
    ADDED = {"by_whom": "TEXT", "pruned": "INTEGER NOT NULL DEFAULT 0"}

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)
        have = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)")}
        for column, decl in self.ADDED.items():
            if column not in have:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {decl}")
        self._conn.commit()
        self._lock = threading.Lock()

    def load(self, node: str) -> list[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, height, kind, source, trust, content, tags, refs, supersedes, topic, "
                "data, ts, prev_hash, hash, by_whom, pruned FROM memories WHERE node = ? ORDER BY seq",
                (node,)).fetchall()
        return [MemoryRecord(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6]), json.loads(r[7]),
                             r[8], r[9], json.loads(r[10]) if r[10] else None, r[11], r[12], r[13],
                             by=r[14], pruned=bool(r[15]))
                for r in rows]

    def append(self, node: str, rec: MemoryRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (node, seq, hash, prev_hash, ts, height, kind, source, trust, content, "
                "tags, refs, supersedes, topic, data, by_whom, pruned) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (node, rec.seq, rec.hash, rec.prev_hash, rec.ts, rec.height, rec.kind, rec.source,
                 rec.trust, rec.content, json.dumps(rec.tags, ensure_ascii=False),
                 json.dumps(rec.refs, ensure_ascii=False), rec.supersedes, rec.topic,
                 json.dumps(rec.data, ensure_ascii=False) if rec.data is not None else None, rec.by))
            self._conn.commit()

    def prune(self, node: str, seqs: list[int]) -> None:
        """Drop the content of these records; hash, link and metadata stay (see Memory.verify_chain)."""
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET content = '', tags = '[]', refs = '[]', data = NULL, pruned = 1 "
                "WHERE node = ? AND seq = ?", [(node, s) for s in seqs])
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
                 data: dict[str, Any] | None = None, by: str | None = None) -> MemoryRecord:
        """by=MASTERMIND for what the model writes through its tools (counted in the quota)."""
        prev = self.records[-1].hash if self.records else GENESIS
        rec = MemoryRecord(len(self.records) + 1, height, kind, source, trust, content,
                           list(tags or []), list(refs or []), supersedes, topic, data,
                           ts=time.time(), prev_hash=prev, by=by)
        rec.hash = sha256_hex(canonical_json(rec.body()))
        self.records.append(rec)
        self._by_hash[rec.hash] = rec
        if self.store:
            self.store.append(self.node_id, rec)
        return rec

    # ---- storage quota ----

    @staticmethod
    def by_mastermind(r: MemoryRecord) -> bool:
        if r.by is not None:
            return r.by == MASTERMIND
        # Records from before `by` existed: notes, web facts and the model's own facts and
        # impressions (Core never writes those kinds with these sources).
        return r.kind == "note" or r.source.startswith("web:") or (
            r.source == "self" and r.kind in ("fact", "relation"))

    @staticmethod
    def size(r: MemoryRecord) -> int:
        return len(r.content.encode()) + (len(json.dumps(r.data, ensure_ascii=False).encode()) if r.data else 0)

    def held(self) -> list[MemoryRecord]:
        """What the Mastermind wrote and still holds: not superseded (older note versions), not forgotten."""
        dead = self.superseded()
        return [r for r in self.records if self.by_mastermind(r) and not r.pruned
                and r.kind != "forget" and r.hash not in dead]

    def usage_bytes(self) -> int:
        """Storage counted against the node's quota: only `held()` records."""
        return sum(self.size(r) for r in self.held())

    def largest(self, n: int = 3) -> str:
        top = sorted(self.held(), key=self.size, reverse=True)[:n]
        return ", ".join(f"#{r.seq} {r.kind}{f' {r.topic!r}' if r.topic else ''} {-(-self.size(r) // 1024)}KB"
                         for r in top) or "(없음)"

    def get(self, seq: int) -> MemoryRecord | None:
        return self.records[seq - 1] if 1 <= seq <= len(self.records) else None

    def forget(self, height: int, target: MemoryRecord, reason: str) -> MemoryRecord:
        """The Mastermind drops one of its records: a forget marker supersedes it (append-only),
        and its content is deleted. Frees quota; the hash chain stays verifiable."""
        marker = self.remember(height, "forget", "self", 1.0, f"#{target.seq} 지움: {reason}"[:300],
                               supersedes=target.hash, by=MASTERMIND)
        self._prune([target])
        return marker

    def compact(self, before_height: int, height: int) -> MemoryRecord | None:
        """Core: fold its own episode records (and superseded ones, e.g. old subscription
        deliveries) written before `before_height` into one digest record; their content is dropped."""
        dead = self.superseded()
        old = [r for r in self.records if r.height < before_height and not r.pruned
               and not self.by_mastermind(r) and r.kind not in _KEEP_KINDS
               and (r.kind == "episode" or r.hash in dead)]
        if not old:
            return None
        kinds: dict[str, int] = {}
        for r in old:
            label = r.content[1:r.content.index("]")] if r.content.startswith("[") and "]" in r.content[:40] \
                else r.source.split(":")[0]
            kinds[label] = kinds.get(label, 0) + 1
        head = (f"블록 {old[0].height}~{old[-1].height}의 Core 기록 {len(old)}건을 요약으로 정리함: "
                + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + ".\n최근:")
        lines, size = [], len(head)
        for r in reversed(old):
            line = f"\n- #{r.seq} {r.content[:120]}"
            if size + len(line) > DIGEST_MAX_CHARS:
                break
            lines.append(line)
            size += len(line)
        digest = self.remember(height, "digest", "core", 1.0, head + "".join(reversed(lines)), ["digest"],
                               data={"seqs": [r.seq for r in old], "from_height": old[0].height,
                                     "to_height": old[-1].height, "kinds": kinds})
        self._prune(old)
        return digest

    def _prune(self, recs: list[MemoryRecord]) -> None:
        for r in recs:
            r.content, r.tags, r.refs, r.data, r.pruned = "", [], [], None, True
        if self.store:
            self.store.prune(self.node_id, [r.seq for r in recs])

    def verify_chain(self) -> bool:
        """Every link, and every hash of a record that still has its content. A pruned record must
        be empty and named by a later (hashed) digest or forget record, so nothing is dropped silently."""
        by_digest: dict[int, int] = {}
        by_forget: dict[str, int] = {}
        for r in self.records:
            if r.kind == "digest" and r.source == "core" and r.data:
                by_digest.update({s: r.seq for s in r.data.get("seqs", [])})
            elif r.kind == "forget" and r.supersedes:
                by_forget[r.supersedes] = r.seq
        prev = GENESIS
        for r in self.records:
            if r.prev_hash != prev:
                return False
            if r.pruned:
                named_by = by_digest.get(r.seq) or by_forget.get(r.hash)
                if r.content or r.data is not None or named_by is None or named_by <= r.seq:
                    return False
            elif sha256_hex(canonical_json(r.body())) != r.hash:
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
            if r.hash in dead or r.pruned:
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
