"""Test questions for the experiment: asks one question from a bank through the Bridge API, the
way an outside client would, so the nodes see some outside demand (operator decision 2026-10-13:
mostly node-to-node questions from research, plus a few of these).

    python -m aichain.feeder setup [--daily-limit 4]   # once: Bridge client "feeder", token in data/feeder_token
    python -m aichain.feeder post                       # what aichain-feeder.timer runs

Each question in the bank is asked again only after all the others have been (state in
data/feeder_state.json). The Bridge reads its client list at start, so restart it after setup.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from aichain.bridge.tokens import DEFAULT_FILE as TOKENS_FILE, add_client

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "deploy" / "test-questions.json"
TOKEN_FILE = ROOT / "data" / "feeder_token"
STATE_FILE = ROOT / "data" / "feeder_state.json"
CLIENT = "feeder"


def load_bank(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))["questions"]
    ids = [q["id"] for q in items]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    for q in items:
        missing = {"id", "title", "body", "acceptance_criteria"} - set(q)
        if missing:
            raise ValueError(f"question {q.get('id')} is missing {sorted(missing)}")
    return items


def pick(bank: list[dict[str, Any]], state: dict[str, Any], rnd: random.Random) -> dict[str, Any]:
    asked = set(state.get("asked", []))
    fresh = [q for q in bank if q["id"] not in asked]
    if not fresh:  # every question has been asked once: start a new round
        state["asked"], fresh = [], list(bank)
    return rnd.choice(fresh)


def post(url: str, token: str, bank_path: Path, state_path: Path, *, client: httpx.Client | None = None,
         rnd: random.Random | None = None) -> int:
    bank = load_bank(bank_path)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    q = pick(bank, state, rnd or random.Random())
    body = {k: q[k] for k in ("title", "body", "acceptance_criteria")}
    body["max_reward_P"] = q.get("max_reward_P", 5)
    body["deadline_minutes"] = q.get("deadline_minutes", 90)
    http = client or httpx.Client(base_url=url, timeout=30)
    r = http.post("/bridge/v1/questions", json=body, headers={"Authorization": f"Bearer {token}"})
    if r.status_code == 429:
        print(f"feeder: daily limit reached, nothing asked ({r.json().get('detail')})")
        return 0
    if r.status_code >= 400:
        print(f"feeder: Bridge refused the question: {r.status_code} {r.text[:300]}", file=sys.stderr)
        return 1
    ref = r.json()["ref"]
    state.setdefault("asked", []).append(q["id"])
    state.setdefault("refs", []).append({"id": q["id"], "ref": ref, "at": int(time.time())})
    state["refs"] = state["refs"][-200:]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(state_path)
    print(f"feeder: asked {q['id']} ({q['title']}) -> {ref}")
    return 0


def setup(daily_limit: int, tokens_file: Path, token_file: Path) -> int:
    if token_file.exists():
        print(f"feeder: {token_file} already exists; nothing to do")
        return 0
    token = add_client(tokens_file, CLIENT, daily_limit)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    print(f"feeder: Bridge client '{CLIENT}' ({daily_limit}/day) created; restart aichain-bridge to load it")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aichain.feeder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="create the Bridge client and store its token (once)")
    s.add_argument("--daily-limit", type=int, default=4)
    p = sub.add_parser("post", help="ask one test question")
    p.add_argument("--url", default="http://127.0.0.1:8780")
    p.add_argument("--bank", type=Path, default=BANK)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    if args.cmd == "setup":
        return setup(args.daily_limit, TOKENS_FILE, TOKEN_FILE)
    if not TOKEN_FILE.exists():
        print(f"feeder: no token at {TOKEN_FILE}; run `python -m aichain.feeder setup` first", file=sys.stderr)
        return 0  # not set up: the timer does nothing rather than fail every few hours
    return post(args.url, TOKEN_FILE.read_text().strip(), args.bank, STATE_FILE)


if __name__ == "__main__":
    sys.exit(main())
