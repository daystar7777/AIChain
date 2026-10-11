"""Runs the network in a background thread and tracks Bridge questions.

The simulation thread is the only writer to the board. Episodes may block on LLM
calls for a long time, so the HTTP side never touches the network directly: it puts
questions into an inbox and reads the per-question records and the network snapshot
this module maintains (each guarded by a short lock).
"""

from __future__ import annotations

import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aichain.board import BountyState
from aichain.common import fmt_P
from aichain.node import REP_PLACEHOLDER
from aichain.sim.network import Network

TERMINAL = {"settled", "no_bounty", "failed"}


@dataclass
class QuestionRecord:
    ref: str
    client: str
    node: str
    title: str
    body: str
    acceptance_criteria: str
    max_reward_P: float | None
    deadline_minutes: int | None
    created_at: float
    status: str = "queued"
    bounty_id: str | None = None
    bounty: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    note: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "status": self.status,
            "node": self.node,
            "title": self.title,
            "created_at": self.created_at,
            "bounty": self.bounty,
            "result": self.result,
            "note": self.note,
        }


class NetworkRuntime:
    def __init__(self, net: Network, block_seconds: float = 5.0, bridge_node: str = "n1") -> None:
        self.net = net
        self.block_seconds = block_seconds
        self.bridge_node = bridge_node
        self._inbox: queue.Queue[QuestionRecord] = queue.Queue()
        self._lock = threading.Lock()
        self._questions: dict[str, QuestionRecord] = {}
        self._by_bounty: dict[str, str] = {}
        self._snapshot: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []
        net.listeners.append(self._on_log)
        self._refresh_snapshot()

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="aichain-sim", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        last_block = time.monotonic()
        failures = 0
        while not self._stop.is_set():
            if self.net.halted:
                # Fail closed: stop producing blocks and running episodes; the API keeps answering.
                self._refresh_snapshot()
                self._stop.wait(1.0)
                continue
            try:
                self._drain_inbox()
                self.net.run_until_idle()
                if time.monotonic() - last_block >= self.block_seconds:
                    last_block = time.monotonic()
                    self.net.advance(1)
                self._refresh_snapshot()
                failures = 0
            except Exception as e:  # keep the network alive; surface the error, back off if it repeats
                failures += 1
                self.errors.append(repr(e))
                self.errors[:] = self.errors[-100:]
                try:
                    self.net.log("runtime", "runtime_error", error=repr(e)[:500], repeated=failures)
                except Exception:
                    pass
                self._stop.wait(min(60.0, 0.5 * 2 ** min(failures, 7)))
                continue
            self._stop.wait(min(0.05, self.block_seconds))

    def _drain_inbox(self) -> None:
        while True:
            try:
                rec = self._inbox.get_nowait()
            except queue.Empty:
                return
            extra: dict[str, Any] = {"ref": rec.ref}
            if rec.max_reward_P is not None:
                extra["max_reward_P"] = rec.max_reward_P
            if rec.deadline_minutes is not None:
                extra["deadline_minutes"] = rec.deadline_minutes
            self.net.bridge_question(rec.node, rec.client, rec.title, rec.body,
                                     rec.acceptance_criteria, **extra)

    # ---- API side ----

    def submit(self, client: str, title: str, body: str, acceptance_criteria: str,
               max_reward_P: float | None = None, deadline_minutes: int | None = None) -> QuestionRecord:
        rec = QuestionRecord(
            ref="q_" + secrets.token_hex(8), client=client, node=self.bridge_node, title=title,
            body=body, acceptance_criteria=acceptance_criteria, max_reward_P=max_reward_P,
            deadline_minutes=deadline_minutes, created_at=time.time())
        with self._lock:
            self._questions[rec.ref] = rec
            self._push(rec, "queued", {})
        self._inbox.put(rec)
        return rec

    def question(self, ref: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._questions.get(ref)
            return None if rec is None else {**rec.public(), "client": rec.client}

    def events_since(self, ref: str, after: int) -> tuple[list[dict[str, Any]], bool]:
        with self._lock:
            rec = self._questions[ref]
            return [e for e in rec.events if e["id"] > after], rec.status in TERMINAL

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot

    # ---- simulation side (runs in the sim thread) ----

    def _push(self, rec: QuestionRecord, status: str, data: dict[str, Any]) -> None:
        rec.status = status
        rec.events.append({"id": len(rec.events) + 1, "event": status, "time": time.time(),
                           "height": self.net.board.height, **data})

    def _bounty_view(self, bounty_id: str) -> dict[str, Any]:
        board = self.net.board
        b = board.bounty(bounty_id)
        return {
            "bounty_id": b.bounty_id, "state": str(b.state),
            "outcome": str(b.outcome) if b.outcome else None,
            "reward": fmt_P(b.reward_mP), "tags": b.tags, "answers": len(b.answers),
            "commit_until": b.commit_until, "reveal_until": b.reveal_until,
            "judge_until": b.judge_until,
        }

    def _result_view(self, bounty_id: str) -> dict[str, Any]:
        board = self.net.board
        b = board.bounty(bounty_id)
        answers = []
        for aid in b.answers:
            a = board.answers[aid]
            if not a.revealed:
                continue
            answers.append({
                "answer_id": a.answer_id, "author": a.author,
                "accepted": aid == b.accepted, "rejected_reason": a.rejected_reason,
                "body": self.net.content.get(a.body_hash),
            })
        return {"outcome": str(b.outcome), "accepted_answer_id": b.accepted, "answers": answers}

    def _on_log(self, e: dict[str, Any]) -> None:
        kind = e["event"]
        with self._lock:
            ref = e.get("ref")
            if ref is None and "bounty_id" in e:
                ref = self._by_bounty.get(e["bounty_id"])
            if ref is None and kind == "tool" and e.get("tool") == "answer":
                ref = self._by_bounty.get((e.get("args") or {}).get("bounty_id", ""))
            rec = self._questions.get(ref) if ref else None
            if rec is None:
                return
            if kind == "episode_start" and e.get("kind") == "bridge_question":
                self._push(rec, "processing", {"node": e["node"]})
            elif kind == "bridge_linked":
                rec.bounty_id = e["bounty_id"]
                self._by_bounty[e["bounty_id"]] = rec.ref
                rec.bounty = self._bounty_view(e["bounty_id"])
                self._push(rec, "bounty_open", {"bounty": rec.bounty})
            elif kind == "episode_end" and e.get("kind") == "bridge_question":
                rec.note = e.get("summary")
                if rec.bounty_id is None:
                    status = "failed" if e.get("stopped") in (
                        "budget", "call_limit", "provider_error") else "no_bounty"
                    self._push(rec, status, {"note": rec.note, "stopped": e.get("stopped")})
            elif kind == "tool" and e.get("tool") == "answer" and rec.bounty_id:
                rec.bounty = self._bounty_view(rec.bounty_id)
                self._push(rec, "answer_committed", {"by": e["node"], "answers": rec.bounty["answers"]})
            elif kind == "reveal_open":
                rec.bounty = self._bounty_view(rec.bounty_id)
                self._push(rec, "revealing", {"bounty": rec.bounty})
            elif kind == "judge_due":
                rec.bounty = self._bounty_view(rec.bounty_id)
                self._push(rec, "judging", {"bounty": rec.bounty})
            elif kind == "bounty_settled":
                rec.bounty = self._bounty_view(rec.bounty_id)
                rec.result = self._result_view(rec.bounty_id)
                self._push(rec, "settled", {"bounty": rec.bounty, "result": rec.result})

    def _refresh_snapshot(self) -> None:
        net = self.net
        board = net.board
        bounties = sorted(board.bounties.values(), key=lambda b: b.created_at, reverse=True)[:10]
        snap = {
            "height": board.height,
            "block_seconds": self.block_seconds,
            "nodes": [
                {"id": n.id, "persona": n.cfg.persona,
                 "balance": fmt_P(board.accounts[n.id].balance_mP),
                 "locked": fmt_P(board.accounts[n.id].locked_mP),
                 "reputation": REP_PLACEHOLDER, "budget_state": str(n.bucket.state)}
                for n in net.nodes.values()
            ],
            "recent_bounties": [
                {"bounty_id": b.bounty_id, "state": str(b.state), "reward": fmt_P(b.reward_mP),
                 "tags": b.tags, "answers": len(b.answers),
                 "title": (net.content.get(b.question_hash) or {}).get("title")}
                for b in bounties
            ],
            "open_bounties": sum(1 for b in board.bounties.values() if b.state == BountyState.OPEN),
            "issued": fmt_P(board.issued_mP),
            "burned": fmt_P(board.burned_mP),
            "llm_calls": net.gateway.calls,
            "halted": net.halted,
            "daily_usage": net.gateway.daily_usage(),
        }
        with self._lock:
            self._snapshot = snap
            # Keep bounty views in records fresh (e.g. FINAL after the dispute window).
            for bid, ref in self._by_bounty.items():
                rec = self._questions[ref]
                rec.bounty = self._bounty_view(bid)
