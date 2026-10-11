"""Operator commands against the running bridge (loopback, operator token).

    python -m aichain.admin disputes [--all]
    python -m aichain.admin show <dispute_id>
    python -m aichain.admin resolve <dispute_id> uphold|reject|void --note "why"

Rulings are signed by the root key inside the network and recorded on the chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AIChain operator commands")
    ap.add_argument("--url", default="http://127.0.0.1:8780")
    ap.add_argument("--token-file", type=Path, default=ROOT / "data" / "admin_token")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("disputes", help="list disputes (open by default)")
    ls.add_argument("--all", action="store_true")
    show = sub.add_parser("show", help="show one dispute with its claim and target")
    show.add_argument("dispute_id")
    res = sub.add_parser("resolve", help="rule on a dispute")
    res.add_argument("dispute_id")
    res.add_argument("ruling", choices=["uphold", "reject", "void"])
    res.add_argument("--note", default="")
    args = ap.parse_args(argv)

    token = args.token_file.read_text().strip()
    client = httpx.Client(base_url=args.url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if args.cmd in ("disputes", "show"):
        r = client.get("/admin/v1/disputes", params={"state": "all" if args.cmd == "show" or args.all else "open"})
        r.raise_for_status()
        items = r.json()
        if args.cmd == "show":
            item = next((d for d in items if d["dispute_id"] == args.dispute_id), None)
            if item is None:
                print("not found", file=sys.stderr)
                return 1
            print(json.dumps(item, ensure_ascii=False, indent=2))
            return 0
        if not items:
            print("(no disputes)")
        for d in items:
            claim = (d.get("claim") or {}).get("claim", "")
            print(f"{d['dispute_id']}  [{d['state']}] {d['target_kind']} {d['target_id']}  "
                  f"{d['opener']} -> {d['defendant']}  bond {d['bond_mP'] / 1000:g}P  "
                  f"deadline #{d['deadline']} (now #{d['height']})\n    {claim[:200]}")
        return 0
    r = client.post(f"/admin/v1/disputes/{args.dispute_id}/resolve",
                    json={"ruling": args.ruling, "note": args.note})
    if r.status_code >= 400:
        print(f"error {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    d = r.json()
    print(f"{d['dispute_id']} -> {d['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
