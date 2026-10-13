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
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from aichain.common import fmt_P, sha256_hex

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
-- The daily report reads one day by time (built once, on the first start after an upgrade).
CREATE INDEX IF NOT EXISTS entries_ts ON entries(ts);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ts REAL NOT NULL,
    height INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_run ON metrics(run_id, height);
CREATE INDEX IF NOT EXISTS metrics_ts ON metrics(ts);
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
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_\-]{30,}"),  # Telegram bot tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
    re.compile(r"(?i)((?:OPENCODE_GO_API_KEY|BRAVE_SEARCH_API_KEY|BRAVE_ANSWER_API_KEY|TELEGRAM_BOT_TOKEN)\s*=\s*)\S+"),
]
# Token counters are not secrets.
_ALLOWED_KEYS = {"input_tokens", "output_tokens", "weighted_tokens", "tokens", "max_output_tokens",
                 "max_episode_tokens"}


_DEAL_BLOCK = re.compile(r'(<data source="deal:[^"]*"[^>]*>\n)(.*?)(\n</data>)', re.S)


def strip_deal_content(value: Any) -> Any:
    """Replace purchased/sold information bodies with a hash (production setting)."""
    if isinstance(value, dict):
        if value.get("name") == "offer_create" and isinstance(value.get("arguments"), str):
            try:  # the model's own tool call carries the seller's text as a JSON string
                args = json.loads(value["arguments"])
                value = {**value, "arguments": json.dumps(strip_deal_content(args), ensure_ascii=False)}
            except ValueError:
                pass
        out = {}
        for k, v in value.items():
            if k == "content" and isinstance(v, str) and value.get("kind") in (None, "info") and "price_P" in value:
                out[k] = f"[deal content withheld: sha256 {sha256_hex(v)[:16]}, {len(v)} chars]"
            else:
                out[k] = strip_deal_content(v)
        return out
    if isinstance(value, list):
        return [strip_deal_content(v) for v in value]
    if isinstance(value, str) and "deal:" in value:
        return _DEAL_BLOCK.sub(lambda m: m.group(1) + f"[deal content withheld: sha256 {sha256_hex(m.group(2))[:16]}, "
                               f"{len(m.group(2))} chars]" + m.group(3), value)
    return value


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
    """Writer side. One instance per process; safe to call from several threads.

    Entries are written as they happen. The viewer's live snapshot (open and recent state only)
    is captured at most every `snapshot_interval` seconds on the network's thread and redacted,
    serialised and written by a background writer, which also runs the prompt pruning; neither
    holds up the thread that produces blocks."""

    METRICS_EVERY_BLOCKS = 60  # 5 minutes at 5 s/block

    PRUNE_EVERY_SECONDS = 6 * 3600
    PRUNE_CHUNK = 2000  # entries per prune transaction

    def __init__(self, path: Path, snapshot_interval: float = 5.0, record_plaintext: bool = True,
                 keep_prompts_days: float = 14.0, background: bool = True) -> None:
        """record_plaintext: keep the bodies of traded information in the journal (development).
        Off: they are replaced by a hash before anything is written.
        keep_prompts_days: full LLM prompts (the bulk of the journal) are dropped from llm_call
        entries older than this; the response, tool calls and token usage stay. 0 keeps them forever.
        background: write snapshots and prune from a writer thread (off: inline, e.g. for tools)."""
        self.keep_prompts_days = keep_prompts_days
        self._last_prune = 0.0
        self._pruned_upto = 0  # entries up to this id have been pruned already
        self.path = path
        self.record_plaintext = record_plaintext
        self._conn = connect(path)
        self._lock = threading.Lock()
        self._snapshot_interval = snapshot_interval
        self._last_snapshot = 0.0
        self.run_id: int | None = None
        self._net: Any = None
        self._balances: dict[str, tuple[int, int]] = {}
        self._last_metrics_height = -1
        self._captured_height = -1
        self._dirty = True
        self._background = background
        self._writer: threading.Thread | None = None
        self._wake = threading.Condition()
        self._pending: tuple[float, dict[str, Any], dict[str, Any] | None] | None = None
        self._stopping = False
        self.write_errors = 0

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
        self._stop_writer()
        try:
            self.write_snapshot(force=True)
        except Exception as e:  # e.g. disk full at shutdown: still mark the run ended
            self._failed(e)
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
        clean = redact(data) if self.record_plaintext else strip_deal_content(redact(data))
        payload = json.dumps(clean, ensure_ascii=False, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT INTO entries (run_id, ts, height, node, kind, episode, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, time.time(), height, node, kind, episode, payload))
            self._conn.commit()
        self._dirty = True

    def prune(self, now: float | None = None) -> int:
        """Drop the prompt messages of old llm_call entries. Returns how many entries changed.
        Works through the table in short id ranges so other writers wait at most one chunk;
        entries are in time order, so it stops at the first one that is still too young."""
        if self.keep_prompts_days <= 0:
            return 0
        cutoff = (now if now is not None else time.time()) - self.keep_prompts_days * 86400
        changed = 0
        while True:
            with self._lock:
                lo = self._pruned_upto
                hi = self._conn.execute("SELECT MAX(id) FROM (SELECT id FROM entries WHERE id > ? "
                                        "ORDER BY id LIMIT ?)", (lo, self.PRUNE_CHUNK)).fetchone()[0]
                if hi is None:
                    break
                cur = self._conn.execute(
                    "UPDATE entries SET data = json_set(json_remove(data, '$.messages'), "
                    "'$.messages_pruned', json('true')) WHERE id > ? AND id <= ? AND kind = 'llm_call' "
                    "AND ts < ? AND json_extract(data, '$.messages') IS NOT NULL", (lo, hi, cutoff))
                changed += cur.rowcount
                young = self._conn.execute("SELECT MIN(id) FROM entries WHERE id > ? AND id <= ? AND ts >= ?",
                                           (lo, hi, cutoff)).fetchone()[0]
                self._conn.commit()
                self._pruned_upto = hi if young is None else young - 1
            if young is not None:
                break
        self._last_prune = time.time()
        return changed

    # ---- attaching to a network ----

    def attach(self, net: Any) -> None:
        """Record every network log entry, every LLM call, and balance changes."""
        self._net = net
        self._balances = {nid: (a.balance_mP, a.locked_mP) for nid, a in net.board.accounts.items()}
        net.listeners.append(self._on_log)
        net.gateway.observers.append(self._on_llm_call)
        self.write_snapshot(force=True)
        if self._background and self._writer is None:
            self._stopping = False
            self._writer = threading.Thread(target=self._write_loop, name="journal-writer", daemon=True)
            self._writer.start()

    def _on_log(self, e: dict[str, Any]) -> None:
        data = {k: v for k, v in e.items() if k not in ("height", "node", "event", "episode")}
        self.record(e["event"], data, node=e.get("node"), height=e.get("height"),
                    episode=e.get("episode"))
        self._check_balances()
        self.tick()
        if self._writer is None and time.time() - self._last_prune >= self.PRUNE_EVERY_SECONDS:
            self.prune()

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

    # ---- snapshots ----

    def tick(self) -> None:
        """Capture the live snapshot if one is due (at most every snapshot_interval, and only
        when something changed). Call it from the thread that owns the network; the runtime
        calls it every loop so the viewer stays current when nothing is logged."""
        net = self._net
        if net is None or self.run_id is None:
            return
        now = time.time()
        if now - self._last_snapshot < self._snapshot_interval:
            return
        if not self._dirty and net.board.height == self._captured_height:
            return
        try:
            snap, metrics = self._capture(force=False)
            if self._writer is None:
                self._write(now, snap, metrics)
                return
        except Exception as e:  # the viewer's snapshot must never stop the network
            self._failed(e)
            return
        with self._wake:
            if metrics is None and self._pending is not None:
                metrics = self._pending[2]  # a metrics row the writer has not stored yet
            self._pending = (now, snap, metrics)
            self._wake.notify()

    def write_snapshot(self, force: bool = False) -> None:
        """Capture and write the snapshot now, on the calling thread."""
        net = self._net
        if net is None or self.run_id is None:
            return
        now = time.time()
        if not force and now - self._last_snapshot < self._snapshot_interval:
            return
        snap, metrics = self._capture(force)
        self._write(now, snap, metrics)

    def _capture(self, force: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
        from aichain.metrics import compute as compute_metrics

        net = self._net
        self._last_snapshot = time.time()
        self._dirty = False
        h = net.board.height
        self._captured_height = h
        metrics = None
        if force or h // self.METRICS_EVERY_BLOCKS != self._last_metrics_height // self.METRICS_EVERY_BLOCKS:
            self._last_metrics_height = h
            metrics = compute_metrics(net)
        return network_snapshot(net), metrics

    def _write(self, ts: float, snap: dict[str, Any], metrics: dict[str, Any] | None) -> None:
        payload = json.dumps(redact(snap), ensure_ascii=False, default=str)
        with self._lock:
            if metrics is not None:
                self._conn.execute("INSERT INTO metrics (run_id, ts, height, data) VALUES (?, ?, ?, ?)",
                                   (self.run_id, ts, metrics["height"], json.dumps(metrics, ensure_ascii=False)))
            self._conn.execute(
                "INSERT INTO snapshots (run_id, ts, data) VALUES (?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET ts = excluded.ts, data = excluded.data",
                (self.run_id, ts, payload))
            self._conn.commit()

    def _write_loop(self) -> None:
        while True:
            with self._wake:
                if self._pending is None and not self._stopping:
                    self._wake.wait(timeout=60.0)
                item, self._pending = self._pending, None
                stopping = self._stopping
            try:
                if item is not None:
                    self._write(*item)
                if not stopping and time.time() - self._last_prune >= self.PRUNE_EVERY_SECONDS:
                    self.prune()
            except Exception as e:  # disk full, database locked: keep going, say so on stderr
                self._failed(e)
            if stopping:
                return

    def _failed(self, e: Exception) -> None:
        self.write_errors += 1
        if self.write_errors <= 5 or self.write_errors % 100 == 0:
            print(f"[aichain] journal snapshot failed ({self.write_errors}x): {type(e).__name__}: {e}",
                  file=sys.stderr)

    def _stop_writer(self) -> None:
        if self._writer is None:
            return
        with self._wake:
            self._stopping = True
            self._wake.notify()
        self._writer.join(timeout=30)
        self._writer = None


def _message_view(m: Any) -> dict[str, Any]:
    view: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        view["tool_calls"] = [{"call_id": c.call_id, "name": c.name, "arguments": c.arguments}
                              for c in m.tool_calls]
    if m.call_id:
        view["call_id"] = m.call_id
        view["name"] = m.name
    return view


SNAPSHOT_RECENT = 20  # closed items per list kept in the live snapshot, besides every open one

_BOUNTY_DONE = {"FINAL", "EXPIRED"}
_PROPOSAL_DONE = {"ACTIVATED", "REJECTED", "EXPIRED", "VETOED"}


def _open_and_recent(items: Any, is_open: Callable[[Any], bool], recent: int) -> list[Any]:
    """Every open item plus the `recent` newest closed ones, in creation order. The full
    history stays in the journal entries (and on the chain); the snapshot is the live view."""
    out, closed = [], 0
    for x in reversed(items):  # dicts keep creation order: newest first
        if is_open(x):
            out.append(x)
        elif closed < recent:
            out.append(x)
            closed += 1
    return out[::-1]


def network_snapshot(net: Any, recent: int = SNAPSHOT_RECENT) -> dict[str, Any]:
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
            "storage_kb": [n.storage_used_kb(), n.storage_quota_kb()],
            "persona_version": n.persona_version,
            "reputation": round(net.reputation(n.id).value, 3),
            "reputation_terms": {k: round(v / 1e6, 3) for k, v in net.reputation(n.id).terms_e6.items()},
            "paused": n.policy.paused,
        })
    bounties = []
    for b in _open_and_recent(board.bounties.values(), lambda b: str(b.state) not in _BOUNTY_DONE, recent):
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
    tools = [t for t in board.tools.values() if t.author != "lib"]
    licenses = Counter((tid, v) for (_, tid, v) in board.licenses)
    return {
        "height": board.height,
        "nodes": nodes,
        "bounties": bounties,
        "supply": {"issued": fmt_P(board.issued_mP), "burned": fmt_P(board.burned_mP),
                   "escrow": fmt_P(board.escrow_mP), "balanced": board.total_supply_check()},
        "llm_calls": net.gateway.calls,
        "disputes": [{"dispute_id": d.dispute_id, "state": d.state, "target_kind": d.target_kind,
                      "target_id": d.target_id, "opener": d.opener, "defendant": d.defendant,
                      "bond": fmt_P(d.bond_mP), "deadline": d.deadline,
                      "claim": (net.content.get(d.claim_hash) or {}).get("claim")}
                     for d in _open_and_recent(board.disputes.values(), lambda d: d.state == "open", recent)],
        "proposals": [{"proposal_id": x.proposal_id, "kind": x.kind, "title": x.title, "proposer": x.proposer,
                       "state": x.state, "changes": x.changes, "votes": x.votes, "tally": x.tally,
                       "vote_until": x.vote_until, "timelock_until": x.timelock_until}
                      for x in _open_and_recent(board.proposals.values(),
                                                lambda x: x.state not in _PROPOSAL_DONE, recent)],
        "params": {k: getattr(board.params, k) for k in ("commit_blocks", "reveal_blocks", "judge_blocks",
                                                          "dispute_blocks", "commit_bond_mP", "block_reward_mP")},
        "tool_requests": [{"request_id": r.request_id, "buyer": r.buyer, "title": r.title, "state": r.state,
                           "reward": fmt_P(r.reward_mP), "deadline": r.deadline,
                           "submissions": len(r.submissions), "winner": r.winner}
                          for r in _open_and_recent(board.tool_requests.values(),
                                                    lambda r: r.state == "OPEN", recent)],
        "offers": [{"offer_id": o.offer_id, "seller": o.seller, "title": o.title, "state": o.state,
                    "price": fmt_P(o.price_mP), "sold": o.sold, "kind": o.kind, "quantity": o.quantity}
                   for o in _open_and_recent(board.offers.values(), lambda o: o.state == "OPEN", recent)],
        "subscriptions": [{"sub_id": x.sub_id, "seller": x.seller, "subscriber": x.subscriber, "state": x.state,
                           "price": fmt_P(x.price_mP), "deliveries": x.deliveries, "paid": x.periods_paid,
                           "refunded": x.periods_refunded}
                          for x in _open_and_recent(board.subscriptions.values(),
                                                    lambda x: x.state == "ACTIVE", recent)],
        "loans": [{"loan_id": x.loan_id, "borrower": x.borrower, "lender": x.lender, "state": x.state,
                   "principal": fmt_P(x.principal_mP), "interest": fmt_P(x.interest_mP),
                   "collateral": fmt_P(x.collateral_mP), "due": x.due}
                  for x in _open_and_recent(board.loans.values(),
                                            lambda x: x.state in ("REQUESTED", "ACTIVE"), recent)],
        "quotas": {n: {"llm_calls": net.gateway.remaining(n, "llm_calls"),
                       "tool_runs": net.gateway.remaining(n, "tool_runs")} for n in net.nodes},
        "tools": [{"tool_id": t.tool_id, "version": t.version, "author": t.author,
                   "price_install": fmt_P(t.price_install_mP), "price_call": fmt_P(t.price_call_mP),
                   "sold": licenses[(t.tool_id, t.version)], "withdrawn": t.withdrawn}
                  for t in _open_and_recent(tools, lambda t: not t.withdrawn, recent)],
        # How many of each exist in total, so the viewer can say what it is not showing.
        "totals": {"bounties": len(board.bounties), "disputes": len(board.disputes),
                   "proposals": len(board.proposals), "tool_requests": len(board.tool_requests),
                   "offers": len(board.offers), "subscriptions": len(board.subscriptions),
                   "loans": len(board.loans), "tools": len(tools)},
        "chain": ({"last_block": board.chain_state.prev_hash, "pending_txs": len(board.chain_state.pending)}
                  if getattr(board, "chain_state", None) else None),
    }
