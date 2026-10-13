"""Operator commands against the running bridge (loopback, operator token).

    python -m aichain.admin disputes [--all]
    python -m aichain.admin show <dispute_id>
    python -m aichain.admin resolve <dispute_id> uphold|reject|void --note "why"
    python -m aichain.admin proposals [--all]
    python -m aichain.admin vote <proposal_id> yes|no|abstain --reason "why"   (operator seat)
    python -m aichain.admin veto <proposal_id> --reason "why"                   (root, timelock only)
    python -m aichain.admin approvals [--all]                      (outflows waiting for the manager)
    python -m aichain.admin approve <approval_id> [--note "..."] / deny <approval_id> [--note "..."]
    python -m aichain.admin persona <node> [--version N]          (history, or show one version)
    python -m aichain.admin rollback <node> <version> --note "why"
    python -m aichain.admin unpause <node> [--note "why"]          (lift a Policy Guard pause)

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
    pl = sub.add_parser("proposals", help="list governance proposals (open by default)")
    pl.add_argument("--all", action="store_true")
    vo = sub.add_parser("vote", help="vote with the operator seat")
    vo.add_argument("proposal_id")
    vo.add_argument("choice", choices=["yes", "no", "abstain"])
    vo.add_argument("--reason", required=True)
    ve = sub.add_parser("veto", help="root veto during the timelock")
    ve.add_argument("proposal_id")
    ve.add_argument("--reason", required=True)
    ap_ = sub.add_parser("approvals", help="node outflows waiting for the manager")
    ap_.add_argument("--all", action="store_true")
    for verb in ("approve", "deny"):
        x = sub.add_parser(verb, help=f"{verb} a pending outflow")
        x.add_argument("approval_id")
        x.add_argument("--note", default="")
    pe = sub.add_parser("persona", help="persona version history of a node")
    pe.add_argument("node")
    pe.add_argument("--version", type=int)
    rb = sub.add_parser("rollback", help="restore an earlier persona version")
    rb.add_argument("node")
    rb.add_argument("version", type=int)
    rb.add_argument("--note", default="")
    up = sub.add_parser("unpause", help="lift a Policy Guard pause after reviewing it")
    up.add_argument("node")
    up.add_argument("--note", default="")
    args = ap.parse_args(argv)

    token = args.token_file.read_text().strip()
    client = httpx.Client(base_url=args.url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if args.cmd == "approvals":
        r = client.get("/admin/v1/approvals", params={"state": "all" if args.all else "PENDING"})
        r.raise_for_status()
        items = r.json()
        if not items:
            print("(no approvals)")
        for a in items:
            left = (a["deadline"] - __import__("time").time()) / 3600
            print(f"{a['approval_id']}  [{a['state']}] {a['node']} {a['tool']} {a['amount_mP'] / 1000:g}P  "
                  f"auto-execute in {left:.1f}h\n    {a['reason'][:200]}")
        return 0
    if args.cmd in ("approve", "deny"):
        r = client.post(f"/admin/v1/approvals/{args.approval_id}", json={"decision": args.cmd, "note": args.note})
        if r.status_code >= 400:
            print(f"error {r.status_code}: {r.text}", file=sys.stderr)
            return 1
        print(f"{args.approval_id} -> {r.json()['state']}")
        return 0
    if args.cmd == "persona":
        r = client.get(f"/admin/v1/persona/{args.node}")
        r.raise_for_status()
        data = r.json()
        if args.version:
            item = next((h for h in data["history"] if h["version"] == args.version), None)
            print(item["text"] if item else "no such version")
            return 0
        print(f"{data['node']}: current v{data['current_version']}")
        for h in data["history"]:
            extra = f" failed={h.get('failed')}" if h.get("failed") else ""
            print(f"  v{h['version']} {h['verdict']} @#{h['height']}{extra}  {h.get('reason', h.get('note', ''))[:120]}")
        return 0
    if args.cmd == "rollback":
        r = client.post(f"/admin/v1/persona/{args.node}/rollback", json={"version": args.version, "note": args.note})
        if r.status_code >= 400:
            print(f"error {r.status_code}: {r.text}", file=sys.stderr)
            return 1
        print(f"{args.node} -> v{r.json()['version']} (restored v{args.version})")
        return 0
    if args.cmd == "unpause":
        r = client.post(f"/admin/v1/nodes/{args.node}/unpause", json={"note": args.note})
        if r.status_code >= 400:
            print(f"error {r.status_code}: {r.text}", file=sys.stderr)
            return 1
        body = r.json()
        print(f"{args.node} unpaused (was: {body['was']})" if "was" in body else body.get("detail", body))
        return 0
    if args.cmd == "proposals":
        r = client.get("/admin/v1/proposals", params={"state": "all" if args.all else "open"})
        r.raise_for_status()
        items = r.json()
        if not items:
            print("(no proposals)")
        for x in items:
            changes = ", ".join(f"{c['key']} {c['from']}->{c['to']}" for c in x["changes"]) or "-"
            votes = ", ".join(f"{k}:{v}" for k, v in x["votes"].items()) or "no votes yet"
            print(f"{x['proposal_id']}  [{x['state']}] {x['kind']} '{x['title']}' by {x['proposer']}\n"
                  f"    changes: {changes}\n    votes: {votes}  (vote until #{x['vote_until']}, "
                  f"timelock until #{x['timelock_until']}, now #{x['height']})")
        return 0
    if args.cmd in ("vote", "veto"):
        path = f"/admin/v1/proposals/{args.proposal_id}/{args.cmd}"
        body = {"choice": args.choice, "reason": args.reason} if args.cmd == "vote" else {"reason": args.reason}
        r = client.post(path, json=body)
        if r.status_code >= 400:
            print(f"error {r.status_code}: {r.text}", file=sys.stderr)
            return 1
        print(f"{args.proposal_id} -> {r.json()['state']}")
        return 0
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
