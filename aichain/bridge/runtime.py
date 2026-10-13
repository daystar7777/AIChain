"""Runs the network in background threads and tracks Bridge questions.

Threading model. The Board is not thread-safe, so every access to the network holds
`BoardLock`. Two threads take turns on it:
- the block thread produces a block every `block_seconds` on a wall-clock schedule, with the
  Core work that goes with it (Network.advance with inline episodes off: events, duties,
  approvals), and runs operator actions and incoming questions. If it could not get the lock
  in time (slow non-LLM work such as a web fetch held it), it produces the missed heights right
  after, so heights keep tracking wall-clock time.
- the episode thread runs the Mastermind episodes. It holds the lock while it builds prompts
  and executes tools, and lets go of it while it waits on the LLM or web provider (the
  gateways' call_guard), which is where nearly all of an episode's time goes. A provider
  outage holding one call for minutes therefore no longer stops the block clock.
The lock is FIFO, so neither thread starves the other, and after a block that leaves work for
an idle episode thread the block thread hands the lock over before the next block: during a
catch-up, events are still acted on between blocks. The HTTP side never touches the network
directly: it puts questions into an inbox and operator actions into a call queue (both run on
the block thread) and reads the per-question records and the snapshot this module maintains
(guarded by a short lock of their own).
"""

from __future__ import annotations

import json
import os
import queue
import secrets
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import Future
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from aichain.board import BountyState
from aichain.common import fmt_P
from aichain.sim.network import Network

TERMINAL = {"settled", "no_bounty", "failed"}
KEEP_SECONDS = 30 * 86400  # question records older than this (and finished) are dropped
SAVE_EVERY_SECONDS = 30.0
MAX_CATCH_UP = 720  # missed blocks produced in one go before the snapshot and inbox are served again
HAND_OVER_SECONDS = 1.0  # how long a block waits for an idle episode thread to pick up new work


class BoardLock:
    """FIFO (ticket) lock around all network access; not reentrant. Threads get it in the order
    they asked, so a thread that releases and asks again goes behind the one already waiting."""

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.Lock())
        self._next = 0
        self._serving = 0
        self._owner: int | None = None

    def acquire(self) -> None:
        with self._cv:
            ticket = self._next
            self._next += 1
            while ticket != self._serving:
                self._cv.wait()
            self._owner = threading.get_ident()

    def release(self) -> None:
        with self._cv:
            self._owner = None
            self._serving += 1
            self._cv.notify_all()

    def held(self) -> bool:
        return self._owner == threading.get_ident()

    def __enter__(self) -> BoardLock:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


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
        # Operator actions; they run on the block thread, holding the board lock.
        self._calls: queue.Queue[tuple[Any, Future]] = queue.Queue()
        self._lock = threading.Lock()
        self._questions: dict[str, QuestionRecord] = {}
        self._by_bounty: dict[str, str] = {}
        self._snapshot: dict[str, Any] = {}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._episode_thread: threading.Thread | None = None
        self.board_lock = BoardLock()
        self._wake = threading.Event()  # work for the episode thread
        self._episodes_idle = threading.Event()  # the episode thread is waiting for work
        self._episode_turn = threading.Event()  # it has taken the lock (or found nothing to do)
        self.errors: list[str] = []
        self.state_path = state_path
        self._last_save = 0.0
        self._load_state()
        net.listeners.append(self._on_log)
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
            for ref in [r for r, q in self._questions.items()
                        if q.status in TERMINAL and now - q.created_at > KEEP_SECONDS]:
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
        if self._threads:
            return
        self.net.inline_episodes = False  # episodes run on their own thread
        self.net.gateway.call_guard = self._provider_wait
        if self.net.web is not None:
            self.net.web.call_guard = self._provider_wait
        self._episode_thread = threading.Thread(target=self._episode_loop, name="aichain-episodes",
                                                daemon=True)
        self._threads = [threading.Thread(target=self._block_loop, name="aichain-blocks", daemon=True),
                         self._episode_thread]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []
        self.net.gateway.call_guard = None
        if self.net.web is not None:
            self.net.web.call_guard = None
        self.net.inline_episodes = True
        self.save_state()

    @contextmanager
    def _provider_wait(self) -> Iterator[None]:
        """Gateways' call_guard: the episode thread lets go of the board while it waits on a
        provider, so blocks keep coming. Blocks made meanwhile are not part of its episode.
        Anything else (an admin action on the block thread) keeps the lock: a block is atomic."""
        if threading.current_thread() is not self._episode_thread or not self.board_lock.held():
            yield
            return
        episode, self.net.current_episode = self.net.current_episode, None
        self.board_lock.release()
        try:
            yield
        finally:
            self.board_lock.acquire()
            self.net.current_episode = episode

    def _block_loop(self) -> None:
        next_block = time.monotonic() + self.block_seconds
        failures = 0
        halt_reported = False
        while not self._stop.is_set():
            if self.net.halted:
                # Fail closed: stop producing blocks and running episodes; the API keeps answering.
                with self.board_lock:
                    if not halt_reported:
                        halt_reported = True
                        try:
                            self.net.manager_send("core", f"⛔ 체인이 멈췄어요: {self.net.halted}")
                        except Exception:
                            pass
                    self._refresh_snapshot()
                self._stop.wait(1.0)
                continue
            try:
                with self.board_lock:
                    self._drain_calls()
                    self._drain_inbox()
                self._hand_over()
                produced = 0  # every block that is due, also the ones missed while the lock was busy
                while time.monotonic() >= next_block and produced < MAX_CATCH_UP and not self._stop.is_set():
                    with self.board_lock:
                        if self.net.halted:
                            break
                        self.net.advance(1)
                    next_block += self.block_seconds
                    produced += 1
                    self._hand_over()
                with self.board_lock:
                    if produced > 1:
                        self.net.log("runtime", "blocks_caught_up", blocks=produced)
                    self._refresh_snapshot()
                    if time.monotonic() - self._last_save >= SAVE_EVERY_SECONDS:
                        self.save_state()
                failures = 0
            except Exception as e:  # keep the network alive; surface the error, back off if it repeats
                failures += 1
                self._error(e, failures)
                self._stop.wait(min(60.0, 0.5 * 2 ** min(failures, 7)))
                continue
            self._stop.wait(max(0.0, min(0.05, next_block - time.monotonic())))

    def _hand_over(self) -> None:
        """After a block (or new questions) left work for an idle episode thread: wake it and let it
        take the lock before the next block, so the event is acted on in time even mid catch-up."""
        if not self._episodes_idle.is_set() or not any(n.has_pending for n in self.net.nodes.values()):
            return  # a running episode thread takes turns through the FIFO lock anyway
        self._episode_turn.clear()
        self._wake.set()
        self._episode_turn.wait(HAND_OVER_SECONDS)

    def _episode_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            self._episodes_idle.set()
            self._wake.wait(0.05)
            self._wake.clear()
            self._episodes_idle.clear()
            if self._stop.is_set() or self.net.halted:
                self._episode_turn.set()
                self._stop.wait(0.5 if self.net.halted else 0)
                continue
            try:
                with self.board_lock:
                    self._episode_turn.set()
                    self.net.run_until_idle()
                failures = 0
            except Exception as e:
                failures += 1
                self._error(e, failures)
                self._stop.wait(min(60.0, 0.5 * 2 ** min(failures, 7)))

    def _error(self, e: Exception, failures: int) -> None:
        self.errors.append(repr(e))
        self.errors[:] = self.errors[-100:]
        try:
            with self.board_lock:
                self.net.log("runtime", "runtime_error", error=repr(e)[:500], repeated=failures)
        except Exception:
            pass

    def call(self, fn: Any, timeout: float = 30.0) -> Any:
        """Run fn(net) on the block thread and return its result (for admin actions)."""
        if not self._threads:  # not started (tests): run inline
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
            elif kind == "duty_deferred" and e.get("kind") == "bridge_question" and rec.bounty_id is None:
                # No LLM capacity right now: the node retries it later instead of dropping it.
                rec.note = f"waiting for LLM capacity; the node will retry ({e.get('reason')})"
                self._push(rec, "queued", {"note": rec.note, "attempts": e.get("attempts")})
            elif kind == "duty_dropped" and e.get("kind") == "bridge_question" and rec.bounty_id is None:
                rec.note = f"gave up: {e.get('reason')}"
                self._push(rec, "failed", {"note": rec.note})
            elif kind == "episode_end" and e.get("kind") == "bridge_question" and e.get("deferred"):
                pass  # reported by duty_deferred
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
                rec.bounty = self._bounty_view(bid)
