"""Content-addressed store. Chain data refers to bodies by hash (principle 3)."""

from __future__ import annotations

from typing import Any

from aichain.common import canonical_json, sha256_hex


class ContentStore:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    @staticmethod
    def hash_of(obj: Any) -> str:
        return sha256_hex(canonical_json(obj))

    def put(self, obj: Any) -> str:
        h = self.hash_of(obj)
        self._items[h] = obj
        return h

    def get(self, h: str) -> Any | None:
        return self._items.get(h)

    def __contains__(self, h: str) -> bool:
        return h in self._items
