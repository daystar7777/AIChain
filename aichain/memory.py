"""Node memory, prototype version (design doc 04, ch. 4).

Append-only records with source and trust. Recall is keyword overlap × trust × recency;
the real service swaps this for embeddings + sqlite-vec without changing the interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[\w가-힣]+")


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


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 1}


class Memory:
    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    def remember(self, height: int, kind: str, source: str, trust: float, content: str,
                 tags: list[str] | None = None, refs: list[str] | None = None) -> MemoryRecord:
        rec = MemoryRecord(len(self.records) + 1, height, kind, source, trust, content,
                           list(tags or []), list(refs or []))
        self.records.append(rec)
        return rec

    def recall(self, query: str, k: int = 5) -> list[MemoryRecord]:
        q = _tokens(query)
        if not q or not self.records:
            return []
        newest = self.records[-1].seq
        scored = []
        for r in self.records:
            overlap = len(q & (_tokens(r.content) | {t.lower() for t in r.tags}))
            if overlap == 0:
                continue
            recency = 1 / (1 + (newest - r.seq) / 50)
            scored.append((overlap * r.trust * recency, r.seq, r))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [r for _, _, r in scored[:k]]
