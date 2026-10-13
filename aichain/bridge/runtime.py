"""Runs the network in a background thread and tracks Bridge questions.

The simulation thread is the only writer to the board. Episodes may block on LLM
calls for a long time, so the HTTP side never touches the network directly: it puts
questions into an inbox and reads the per-question records and the network snapshot
this module maintains (each guarded by a short lock).
"""

from __future__ import annotations

import json
import os
import queue
import secrets
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import Future
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aichain.board import BountyState
from aichain.common import fmt_P
from aichain.sim.network import Network

TERMINAL = {"settled", "no_bounty", "failed"}
KEEP_SECONDS = 30 * 86400  # question records older than this (and finished) are dropped
MAX_QUESTIONS = 2000  # and beyond this many, the oldest finished ones go first
SAVE_EVERY_SECONDS = 30.0


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
    def __init__(self, net: Network, block_seconds: float = 5.0, bridge_node: str = "n1",
                 state_path: Path | None = None) -> None:
        self.net = net
        self.block_seconds = block_seconds
        self.bridge_node = bridge_node
        self._inbox: queue.Queue[QuestionRecord] = queue.Queue()
        # Operator actions that must run on the simulation thread (it owns the board).
        self._calls: queue.Queue[tuple[Any, Future]] = queue.Queue()
        self._lock = threading.Lock()
        self._questions: dict[str, QuestionRecord] = {}
        self._by_bounty: dict[str, str] = {}
        self._snapshot: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []
        self.state_path = state_path
        self._last_save = 0.0
        # Called on the simulation thread every loop (e.g. the journal's snapshot timer).
        self.tick_hooks: list[Any] = []
        # Halted chain: after this many seconds call on_fatal(reason) (the server exits non-zero
        # so systemd restarts it and replays the log). None: stay up, refusing work (tests).
        self.halt_exit_seconds: float | None = None
        self.on_fatal: Any = None
        # time.monotonic() of the last sign of life: a loop turn, a log entry, an LLM call.
        # The systemd watchdog pings only while this is recent (aichain.ops.Watchdog).
        self.progress_at = 0.0
        self._load_state()
        net.listeners.append(self._on_log)
        net.gateway.observers.append(lambda *_: self._beat())
        self._refresh_snapshot()

    # ---- persistence: Bridge question records and Policy Guard history survive restarts ----

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        for d in raw.get("questions", []):
            rec = QuestionRecord(**d)
            if rec.status not in TERMINAL and rec.bounty_id is None:
                # Its episode was lost with the old process; tell the client instead of hanging.
                rec.note = "the network restarted before this question became a bounty; please ask again"
                rec.status = "failed"
                rec.events.append({"id": len(rec.events) + 1, "event": "failed", "time": time.time(),
                                   "height": self.net.board.height, "note": rec.note})
            self._questions[rec.ref] = rec
            if rec.bounty_id and rec.bounty_id in self.net.board.bounties:
                self._by_bounty[rec.bounty_id] = rec.ref
        for node_id, d in raw.get("policy", {}).items():
            if node_id in self.net.nodes:
                self.net.nodes[node_id].policy.restore(d)

    def save_state(self) -> None:
        if self.state_path is None:
            return
        now = time.time()
        with self._lock:
            done = [r for r, q in self._questions.items() if q.status in TERMINAL]
            excess = max(0, len(self._questions) - MAX_QUESTIONS)
            for ref in done[:excess] + [r for r in done[excess:]
                                        if now - self._questions[r].created_at > KEEP_SECONDS]:
                q = self._questions.pop(ref)
                if q.bounty_id:
                    self._by_bounty.pop(q.bounty_id, None)
            data = {"questions": [asdict(q) for q in self._questions.values()],
                    "policy": {n.id: n.policy.dump() for n in self.net.nodes.values()}}
            text = json.dumps(data, ensure_ascii=False)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(self.state_path)
        self._last_save = time.monotonic()

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
        self.save_state()

    def _beat(self) -> None:
        self.progress_at = time.monotonic()

    def _loop(self) -> None:
        last_block = time.monotonic()
        failures = 0
        halted_at: float | None = None
        while not self._stop.is_set():
            self._beat()
            if self.net.halted:
                # Fail closed: stop producing blocks and running episodes; the API keeps answering
                # until the grace period ends, then the process exits and systemd restarts it.
                exit_in = self.halt_exit_seconds
                if halted_at is None:
                    halted_at = time.monotonic()
                    try:
                        self.net.manager_send("core", f"⛔ 체인이 멈췄어요: {self.net.halted}" + (
                            f"\n{exit_in:.0f}초 뒤 프로세스를 끝내고 systemd가 다시 시작합니다(체인 로그 재생)."
                            if exit_in is not None else ""))
                        self.net.log("runtime", "halted", reason=self.net.halted, exit_in=exit_in)
                    except Exception:
                        pass
                elif exit_in is not None and time.monotonic() - halted_at >= exit_in and self.on_fatal:
                    fatal, self.on_fatal = self.on_fatal, None  # once
                    fatal(self.net.halted)
                self._refresh_snapshot()
                self._stop.wait(1.0)
                continue
            try:
                self._drain_calls()
                self._drain_inbox()
                self.net.run_until_idle()
                if time.monotonic() - last_block >= self.block_seconds:
                    last_block = time.monotonic()
                    self.net.advance(1)
                self._refresh_snapshot()
                for hook in self.tick_hooks:
                    hook()
                if time.monotonic() - self._last_save >= SAVE_EVERY_SECONDS:
                    try:
                        self.save_state()
                    except OSError as e:  # disk full: keep producing blocks, try again in 30 s
                        self._last_save = time.monotonic()
                        self.net.log("runtime", "state_save_failed", error=repr(e)[:300])
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

    def call(self, fn: Any, timeout: float = 30.0) -> Any:
        """Run fn(net) on the simulation thread and return its result (for admin actions)."""
        if self._thread is None:  # not started (tests): run inline
            return fn(self.net)
        fut: Future = Future()
        self._calls.put((fn, fut))
        return fut.result(timeout=timeout)

    def _drain_calls(self) -> None:
        while True:
            try:
                fn, fut = self._calls.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn(self.net))
            except Exception as e:
                fut.set_exception(e)

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
        self._beat()
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
                 "reputation": round(net.reputation(n.id).value, 3), "budget_state": str(n.bucket.state)}
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
                if rec.bounty and rec.bounty["state"] in ("FINAL", "EXPIRED"):
                    continue  # terminal: the view cannot change any more
                rec.bounty = self._bounty_view(bid)
