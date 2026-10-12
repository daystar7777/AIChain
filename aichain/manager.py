"""Node asset management with a human manager (노드 관리인).

The Mastermind manages its node's points on its own; only *outflows* wait for the manager:
every transfer to another account, and any single spend above the threshold (default 100P or
20% of the available balance). The manager approves or denies (admin API, CLI or Telegram).
With no answer before the timeout (default 72 hours, never less) Core executes the action,
re-checking Policy Guard at that moment, so nodes are not stuck on an absent human.

This applies to on-chain points only. Real external assets must never auto-execute on timeout.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

MIN_TIMEOUT_HOURS = 72.0


@dataclass
class ManagerPolicy:
    approval_threshold_mP: int = 100_000
    approval_ratio_bp: int = 2000  # 20% of available balance
    timeout_hours: float = MIN_TIMEOUT_HOURS
    transfers_need_approval: bool = True

    def __post_init__(self) -> None:
        if self.timeout_hours < MIN_TIMEOUT_HOURS:
            raise ValueError(f"approval timeout must be at least {MIN_TIMEOUT_HOURS:g} hours")

    def needs_approval(self, tool: str, amount_mP: int, available_mP: int) -> bool:
        if tool == "transfer" and self.transfers_need_approval:
            return True
        return amount_mP > self.approval_threshold_mP or amount_mP * 10_000 > available_mP * self.approval_ratio_bp


@dataclass
class Approval:
    approval_id: str
    node: str
    tool: str
    args: str  # the exact JSON arguments to execute
    amount_mP: int
    reason: str
    created: float
    deadline: float
    state: str = "PENDING"  # PENDING APPROVED DENIED EXECUTED FAILED
    decided_by: str | None = None
    note: str = ""
    result: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)


class ApprovalBook:
    def __init__(self, policy: ManagerPolicy | None = None, path: Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.policy = policy or ManagerPolicy()
        self.path = path
        self.clock = clock
        self.items: dict[str, Approval] = {}
        self.listeners: list[Callable[[str, Approval], None]] = []  # (event, approval)
        self._lock = threading.Lock()
        if path is not None and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.items = {k: Approval(**v) for k, v in raw.items()}

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.items.items()}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def _emit(self, event: str, a: Approval) -> None:
        for fn in self.listeners:
            try:
                fn(event, a)
            except Exception:
                pass

    def request(self, node: str, tool: str, args: str, amount_mP: int, reason: str) -> Approval:
        now = self.clock()
        a = Approval("ap_" + secrets.token_hex(4), node, tool, args, amount_mP, reason[:500], now,
                     now + self.policy.timeout_hours * 3600)
        with self._lock:
            self.items[a.approval_id] = a
            self._save()
        self._emit("requested", a)
        return a

    def decide(self, approval_id: str, approve: bool, by: str, note: str = "") -> Approval:
        with self._lock:
            a = self.items.get(approval_id)
            if a is None or a.state != "PENDING":
                raise KeyError("no pending approval with that id")
            a.state, a.decided_by, a.note = ("APPROVED" if approve else "DENIED"), by, note[:500]
            a.history.append({"ts": self.clock(), "event": a.state, "by": by})
            self._save()
        self._emit("decided", a)
        return a

    def due(self) -> list[Approval]:
        """Approved ones, and pending ones whose timeout passed (executed by Core)."""
        now = self.clock()
        out = []
        with self._lock:
            for a in self.items.values():
                if a.state == "APPROVED" or (a.state == "PENDING" and now >= a.deadline):
                    if a.state == "PENDING":
                        a.decided_by = "timeout"
                        a.history.append({"ts": now, "event": "TIMEOUT"})
                    out.append(a)
        return out

    def finish(self, a: Approval, ok: bool, result: str) -> None:
        with self._lock:
            a.state, a.result = ("EXECUTED" if ok else "FAILED"), result[:1000]
            a.history.append({"ts": self.clock(), "event": a.state})
            self._save()
        self._emit("finished", a)

    def pending(self, node: str | None = None) -> list[Approval]:
        return [a for a in self.items.values() if a.state == "PENDING" and (node is None or a.node == node)]
