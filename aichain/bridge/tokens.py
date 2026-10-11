"""Bridge client tokens (04 doc 7: per-client API tokens issued by the operator).

Only SHA-256 hashes are stored, in a JSON file:
    {"clients": {"<name>": {"sha256": "<hex>", "daily_limit": 20}}}

    python -m aichain.bridge.tokens new <client> [--daily-limit N] [--file PATH]
prints a new token once; it cannot be recovered later.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from aichain.common import sha256_hex

DEFAULT_FILE = Path("data/bridge_tokens.json")


@dataclass
class Client:
    name: str
    sha256: str
    daily_limit: int


class TokenStore:
    def __init__(self, clients: dict[str, Client]) -> None:
        self.clients = clients
        self._usage: dict[tuple[str, str], int] = {}

    @classmethod
    def load(cls, path: Path) -> TokenStore:
        if not path.exists():
            return cls({})
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls({
            name: Client(name, c["sha256"], int(c.get("daily_limit", 20)))
            for name, c in raw.get("clients", {}).items()
        })

    def authenticate(self, token: str) -> Client | None:
        digest = sha256_hex(token)
        found = None
        for c in self.clients.values():
            if hmac.compare_digest(c.sha256, digest):
                found = c
        return found

    def take_quota(self, client: Client) -> bool:
        """Count one question against today's (UTC) limit. False when exhausted."""
        key = (client.name, time.strftime("%Y-%m-%d", time.gmtime()))
        used = self._usage.get(key, 0)
        if used >= client.daily_limit:
            return False
        self._usage[key] = used + 1
        return True


def add_client(path: Path, name: str, daily_limit: int) -> str:
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"clients": {}}
    token = "acb_" + secrets.token_urlsafe(32)
    raw.setdefault("clients", {})[name] = {"sha256": sha256_hex(token), "daily_limit": daily_limit}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    return token


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Manage Bridge API client tokens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new", help="issue (or replace) a token for a client")
    new.add_argument("client")
    new.add_argument("--daily-limit", type=int, default=20)
    new.add_argument("--file", type=Path, default=DEFAULT_FILE)
    ls = sub.add_parser("list", help="list clients")
    ls.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = ap.parse_args(argv)
    if args.cmd == "new":
        token = add_client(args.file, args.client, args.daily_limit)
        print(token)
    else:
        for c in TokenStore.load(args.file).clients.values():
            print(f"{c.name}\tdaily_limit={c.daily_limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
