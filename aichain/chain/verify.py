"""Check a chain log without touching it: replay a copy and report what was verified.

    python -m aichain.chain.verify data/chain/chain-v2.db [--periods short|design] [--full]

Safe while the Bridge is running (it reads through SQLite's backup API into a temporary copy).
--full replays from genesis instead of the newest snapshot. Exit code 0 = verifies, 3 = does not.
"""

from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from aichain.board import Board, Params
from aichain.chain.store import ChainStore, ReplayMismatch, replay


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aichain.chain.verify")
    ap.add_argument("db", type=Path)
    ap.add_argument("--periods", choices=["short", "design"], default="short",
                    help="the --periods the Bridge runs with (the chain starts from these parameters)")
    ap.add_argument("--full", action="store_true", help="replay from genesis, ignoring snapshots")
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"no chain log at {args.db}", file=sys.stderr)
        return 2
    from aichain.sim.factory import SHORT_PARAMS
    params = replace(SHORT_PARAMS) if args.periods == "short" else Params()
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "chain.db"
        src = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        src.backup(dst)
        src.close(), dst.close()
        board = Board(params)
        started = time.monotonic()
        try:
            stats = replay(board, ChainStore(copy), from_snapshot=not args.full)
        except ReplayMismatch as e:
            print(f"does not verify: {e}", file=sys.stderr)
            return 3
        stats["seconds"] = round(time.monotonic() - started, 1)
        stats["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        stats["rules"] = board.rules
        stats["supply_ok"] = board.total_supply_check()
        stats["escrow_ok"] = board.escrow_obligations() == board.escrow_mP
        print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
