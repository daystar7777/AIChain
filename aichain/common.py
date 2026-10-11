"""Shared helpers: point units, hashing, canonical JSON, <data> escaping."""

from __future__ import annotations

import hashlib
import json
from typing import Any

MP_PER_P = 1000


def to_mP(points: float) -> int:
    return int(round(points * MP_PER_P))


def fmt_P(mP: int) -> str:
    whole, frac = divmod(abs(mP), MP_PER_P)
    sign = "-" if mP < 0 else ""
    return f"{sign}{whole}P" if frac == 0 else f"{sign}{whole}.{frac:03d}".rstrip("0") + "P"


def canonical_json(obj: Any) -> str:
    # Approximation of RFC 8785 (JCS) good enough for the prototype: sorted keys,
    # no whitespace, integers only in chain data.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def escape_data(text: str) -> str:
    """Escape text placed inside a <data> block so it cannot fake tags."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def data_block(source: str, text: str, trust: float | None = None) -> str:
    trust_attr = f' trust="{trust:.2f}"' if trust is not None else ""
    return f'<data source="{escape_data(source)}"{trust_attr}>\n{escape_data(text)}\n</data>'
