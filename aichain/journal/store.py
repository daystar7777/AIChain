"""Journal: a persistent, per-run record of everything the network does, for watching
the prototype during development.

SQLite (WAL) so the read-only web viewer can follow a run while it is being written.
Every value is passed through `redact` first: API keys, bearer tokens and
Authorization headers never reach the database.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from aichain.common import fmt_P

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    label TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ts REAL NOT NULL,
    height INTEGER,
    node TEXT,
    kind TEXT NOT NULL,
    episode TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_run ON entries(run_id, id);
CREATE INDEX IF NOT EXISTS entries_episode ON entries(run_id, episode);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ts REAL NOT NULL,
    height INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_run ON metrics(run_id, height);
CREATE TABLE IF NOT EXISTS snapshots (
    run_id INTEGER PRIMARY KEY REFERENCES runs(id),
    ts REAL NOT NULL,
    data TEXT NOT NULL
);
"""

REDACTED = "[REDACTED]"
_SECRET_KEYS = re.compile(r"(authorization|api[_-]?key|secret|password|passwd|token)$", re.I)
_SECRET_VALUES = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI-style keys (OpenCode Go uses these)
    re.compile(r"\bacb_[A-Za-z0-9_\-]{16,}"),  # Bridge client tokens
    re.compile(r"\bBSA[A-Za-z0-9_\-]{20,}"),  # Brave Search API keys
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
    re.compile(r"(?i)((?:OPENCODE_GO|BRAVE_SEARCH|BRAVE_ANSWER)_API_KEY\s*=\s*)\S+"),
]
# Token counters are not secrets.
_ALLOWED_KEYS = {"input_tokens", "output_tokens", "weighted_tokens", "tokens", "max_output_tokens",
                 "max_episode_tokens"}


def redact(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and k not in _ALLOWED_KEYS and _SECRET_KEYS.search(k):
                out[k] = REDACTED
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in _SECRET_VALUES:
            value = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + REDACTED, value)
        return value
    return value


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


class Journal:
    """Writer side. One instance per process; safe to call from several threads."""

    METRICS_EVERY_BLOCKS = 60  # 5 minutes at 5 s/block

    def __init__(self, path: Path, snapshot_interval: float = 0.5) -> None:
        self.path = path
        self._conn = connect(path)
        self._lock = threading.Lock()
        self._snapshot_interval = snapshot_interval
        self._last_snapshot = 0.0
        self.run_id: int | None = None
        self._net: Any = None
        self._balances: dict[str, tuple[int, int]] = {}
        self._last_metrics_height = -1

    # ---- runs ----

    def start_run(self, label: str, meta: dict[str, Any] | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs (started_at, label, meta) VALUES (?, ?, ?)",
                (time.time(), label, json.dumps(redact(meta or {}), ensure_ascii=False)))
            self._conn.commit()
            self.run_id = cur.lastrowid
        return self.run_id

    def end_run(self) -> None:
        if self.run_id is None:
            return
        self.write_snapshot(force=True)
        with self._lock:
            self._conn.execute("UPDATE runs SET ended_at = ? WHERE id = ?", (time.time(), self.run_id))
            self._conn.commit()

    def close(self) -> None:
        self.end_run()
        with self._lock:
            self._conn.close()

    # ---- entries ----

    def record(self, kind: str, data: dict[str, Any], *, node: str | None = None,
               height: int | None = None, episode: str | None = None) -> None:
        if self.run_id is None:
            raise RuntimeError("start_run first")
        payload = json.dumps(redact(data), ensure_ascii=False, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT INTO entries (run_id, ts, height, node, kind, episode, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, time.time(), height, node, kind, episode, payload))
            self._conn.commit()

    # ---- attaching to a network ----

    def attach(self, net: Any) -> None:
        """Record every network log entry, every LLM call, and balance changes."""
        self._net = net
        self._balances = {nid: (a.balance_mP, a.locked_mP) for nid, a in net.board.accounts.items()}
        net.listeners.append(self._on_log)
        net.gateway.observers.append(self._on_llm_call)
        self.write_snapshot(force=True)

    def _on_log(self, e: dict[str, Any]) -> None:
        data = {k: v for k, v in e.items() if k not in ("height", "node", "event", "episode")}
        self.record(e["event"], data, node=e.get("node"), height=e.get("height"),
                    episode=e.get("episode"))
        self._check_balances()
        self.write_snapshot()

    def _on_llm_call(self, request: Any, response: Any, info: dict[str, Any]) -> None:
        net = self._net
        data: dict[str, Any] = {
            **info,
            "model": request.model,
            "temperature": request.temperature,
            "messages": [_message_view(m) for m in request.messages],
            "tools": [t.name for t in request.tools],
        }
        if response is not None:
            data["response"] = {
                "text": response.text,
                "tool_calls": [{"call_id": c.call_id, "name": c.name, "arguments": c.arguments}
                               for c in response.tool_calls],
                "usage": {"input_tokens": response.usage.input_tokens,
                          "output_tokens": response.usage.output_tokens},
            }
        self.record("llm_call", data, node=request.node_id,
                    height=net.board.height if net else None,
                    episode=net.current_episode if net else None)

    def _check_balances(self) -> None:
        net = self._net
        for nid, a in net.board.accounts.items():
            new = (a.balance_mP, a.locked_mP)
            old = self._balances.get(nid)
            if old != new:
                self._balances[nid] = new
                if old is not None:
                    self.record("balance", {
                        "balance": fmt_P(new[0]), "locked": fmt_P(new[1]),
                        "delta": fmt_P(new[0] - old[0]), "balance_mP": new[0], "locked_mP": new[1],
                    }, node=nid, height=net.board.height, episode=net.current_episode)

    def write_snapshot(self, force: bool = False) -> None:
        net = self._net
        if net is None or self.run_id is None:
            return
        now = time.time()
        if not force and now - self._last_snapshot < self._snapshot_interval:
            return
        self._last_snapshot = now
        snap = network_snapshot(net)
        h = net.board.height
        if force or h // self.METRICS_EVERY_BLOCKS != self._last_metrics_height // self.METRICS_EVERY_BLOCKS:
            self._last_metrics_height = h
            with self._lock:
                self._conn.execute("INSERT INTO metrics (run_id, ts, height, data) VALUES (?, ?, ?, ?)",
                                   (self.run_id, now, h, json.dumps(snap["metrics"], ensure_ascii=False)))
                self._conn.commit()
        with self._lock:
            self._conn.execute(
                "INSERT INTO snapshots (run_id, ts, data) VALUES (?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET ts = excluded.ts, data = excluded.data",
                (self.run_id, now, json.dumps(redact(snap), ensure_ascii=False, default=str)))
            self._conn.commit()


def _message_view(m: Any) -> dict[str, Any]:
    view: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        view["tool_calls"] = [{"call_id": c.call_id, "name": c.name, "arguments": c.arguments}
                              for c in m.tool_calls]
    if m.call_id:
        view["call_id"] = m.call_id
        view["name"] = m.name
    return view


def network_snapshot(net: Any) -> dict[str, Any]:
    board = net.board
    nodes = []
    for n in net.nodes.values():
        a = board.accounts[n.id]
        nodes.append({
            "id": n.id, "persona": n.cfg.persona, "model": n.cfg.model,
            "balance": fmt_P(a.balance_mP), "locked": fmt_P(a.locked_mP),
            "available": fmt_P(a.available_mP),
            "budget_state": str(n.bucket.state), "budget_remaining": int(n.bucket.tokens),
            "budget_capacity": n.bucket.capacity,
            "memories": len(n.memory.records),
            "persona_version": n.persona_version,
            "reputation": round(net.reputation(n.id).value, 3),
            "reputation_terms": {k: round(v / 1e6, 3) for k, v in net.reputation(n.id).terms_e6.items()},
            "paused": n.policy.paused,
        })
    bounties = []
    for b in board.bounties.values():
        q = net.content.get(b.question_hash) or {}
        answers = []
        for aid in b.answers:
            ans = board.answers[aid]
            answers.append({
                "answer_id": aid, "author": ans.author, "revealed": ans.revealed,
                "accepted": aid == b.accepted, "rejected_reason": ans.rejected_reason,
                "body": net.content.get(ans.body_hash) if ans.body_hash else None,
            })
        bounties.append({
            "bounty_id": b.bounty_id, "asker": b.asker, "state": str(b.state),
            "outcome": str(b.outcome) if b.outcome else None, "reward": fmt_P(b.reward_mP),
            "tags": b.tags, "created_at": b.created_at, "commit_until": b.commit_until,
            "reveal_until": b.reveal_until, "judge_until": b.judge_until,
            "settled_at": b.settled_at, "question": q, "answers": answers,
        })
    from aichain.metrics import compute as compute_metrics

    return {
        "height": board.height,
        "metrics": compute_metrics(net),
        "nodes": nodes,
        "bounties": bounties,
        "supply": {"issued": fmt_P(board.issued_mP), "burned": fmt_P(board.burned_mP),
                   "escrow": fmt_P(board.escrow_mP), "balanced": board.total_supply_check()},
        "llm_calls": net.gateway.calls,
        "disputes": [{"dispute_id": d.dispute_id, "state": d.state, "target_kind": d.target_kind,
                      "target_id": d.target_id, "opener": d.opener, "defendant": d.defendant,
                      "bond": fmt_P(d.bond_mP), "deadline": d.deadline,
                      "claim": (net.content.get(d.claim_hash) or {}).get("claim")}
                     for d in board.disputes.values()],
        "proposals": [{"proposal_id": x.proposal_id, "kind": x.kind, "title": x.title, "proposer": x.proposer,
                       "state": x.state, "changes": x.changes, "votes": x.votes, "tally": x.tally,
                       "vote_until": x.vote_until, "timelock_until": x.timelock_until}
                      for x in board.proposals.values()],
        "params": {k: getattr(board.params, k) for k in ("commit_blocks", "reveal_blocks", "judge_blocks",
                                                          "dispute_blocks", "commit_bond_mP", "block_reward_mP")},
        "tool_requests": [{"request_id": r.request_id, "buyer": r.buyer, "title": r.title, "state": r.state,
                           "reward": fmt_P(r.reward_mP), "deadline": r.deadline,
                           "submissions": len(r.submissions), "winner": r.winner}
                          for r in board.tool_requests.values()],
        "offers": [{"offer_id": o.offer_id, "seller": o.seller, "title": o.title, "state": o.state,
                    "price": fmt_P(o.price_mP), "sold": o.sold, "kind": o.kind, "quantity": o.quantity}
                   for o in board.offers.values()],
        "subscriptions": [{"sub_id": x.sub_id, "seller": x.seller, "subscriber": x.subscriber, "state": x.state,
                           "price": fmt_P(x.price_mP), "deliveries": x.deliveries, "paid": x.periods_paid,
                           "refunded": x.periods_refunded} for x in board.subscriptions.values()],
        "loans": [{"loan_id": x.loan_id, "borrower": x.borrower, "lender": x.lender, "state": x.state,
                   "principal": fmt_P(x.principal_mP), "interest": fmt_P(x.interest_mP),
                   "collateral": fmt_P(x.collateral_mP), "due": x.due} for x in board.loans.values()],
        "quotas": {n: {"llm_calls": net.gateway.remaining(n, "llm_calls"),
                       "tool_runs": net.gateway.remaining(n, "tool_runs")} for n in net.nodes},
        "tools": [{"tool_id": t.tool_id, "version": t.version, "author": t.author,
                   "price_install": fmt_P(t.price_install_mP), "price_call": fmt_P(t.price_call_mP),
                   "sold": sum(1 for (_, tid, v) in board.licenses if tid == t.tool_id and v == t.version),
                   "withdrawn": t.withdrawn} for t in board.tools.values()],
        "chain": ({"last_block": board.chain_state.prev_hash, "pending_txs": len(board.chain_state.pending)}
                  if getattr(board, "chain_state", None) else None),
    }
