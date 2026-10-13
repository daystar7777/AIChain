"""Duties are retried, not dropped (design doc 04, 3.1 and 6.1).

A duty event (priority ≤ DUTY_MAX_PRIORITY: judging, disputes and reviews, votes, operator and
Bridge questions, answering bounties) that could not get its LLM calls — token budget, daily
call cap, a provider outage — waits here instead of being lost. Core looks at the list every
block and puts an event back on the node's queue once the gateway would accept its calls again,
as long as the duty is still due (the bounty still judging, the vote still open, ...). Waiting
costs no LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aichain.board import SETTLED_STATES, BountyState
from aichain.gateways.llm.base import is_duty

if TYPE_CHECKING:
    from aichain.node import Node, QueuedEvent

# Episode stops that mean "no capacity" rather than "handled".
CAPACITY_STOPS = {"budget", "call_limit", "provider_error"}
MAX_WAIT_BLOCKS = 17_280  # a duty still waiting after a day is given up (and logged)
PROVIDER_BACKOFF_BLOCKS = 60  # 5 minutes between attempts after a provider outage
MAX_PROVIDER_ATTEMPTS = 6
# Events whose own on-chain state shows whether the work is done, so they can be retried even
# after the cut-short episode already wrote something. Others are retried only if it wrote nothing.
SELF_CHECKING = {"judge_due", "proposal_open", "bounty_new"}


@dataclass
class Waiting:
    kind: str
    payload: dict[str, Any]  # carries "retry": {"since": height, "attempts": n}
    priority: int
    not_before: int
    reason: str


class DutyRetry:
    def __init__(self, node: Node) -> None:
        self.node = node
        self.waiting: list[Waiting] = []

    def refused(self, ev: QueuedEvent, reason: str) -> bool:
        """The event was not started (budget state). True if it will be retried."""
        return self._wait(ev, reason, provider=False)

    def after_episode(self, ev: QueuedEvent, stopped: str, summary: str | None, writes: int,
                      provider_retryable: bool) -> bool:
        """The episode ended without capacity. True if the duty will be retried."""
        if stopped not in CAPACITY_STOPS or (stopped == "provider_error" and not provider_retryable):
            return False
        if writes and ev.kind not in SELF_CHECKING:
            return False  # it already acted; running it again could repeat that
        return self._wait(ev, summary or stopped, provider=stopped == "provider_error")

    def release(self) -> None:
        """Core, every block: queue waiting duties again once their calls would be accepted."""
        if not self.waiting:
            return
        net, h = self.node.net, self.node.net.board.height
        keep = []
        for w in self.waiting:
            if not self.still_due(w.kind, w.payload):
                self._drop(w.kind, w.payload, "no longer due")
            elif h - w.payload["retry"]["since"] > MAX_WAIT_BLOCKS:
                self._drop(w.kind, w.payload, f"still no capacity after a day: {w.reason}")
            elif h >= w.not_before and net.gateway.can_call(self.node.id, w.priority):
                self.node.enqueue(w.kind, w.payload)
            else:
                keep.append(w)
        self.waiting = keep

    def still_due(self, kind: str, payload: dict[str, Any]) -> bool:
        board, me = self.node.net.board, self.node.id
        if kind in ("judge_due", "bounty_new", "review_settled"):
            b = board.bounties.get(payload.get("bounty_id", ""))
            if b is None:
                return False
            if kind == "judge_due":
                return b.asker == me and b.state == BountyState.JUDGING and b.accepted is None
            if kind == "bounty_new":
                return b.state == BountyState.OPEN and not any(board.answers[a].author == me for a in b.answers)
            return b.state in SETTLED_STATES  # the dispute window is still open
        if kind == "proposal_open":
            prop = board.proposals.get(payload.get("proposal_id", ""))
            return prop is not None and prop.state == "VOTING" and me not in prop.votes
        if kind == "tool_request_new":
            req = board.tool_requests.get(payload.get("request_id", ""))
            return req is not None and req.state == "OPEN"
        return True  # operator and Bridge questions: until MAX_WAIT_BLOCKS

    def _wait(self, ev: QueuedEvent, reason: str, *, provider: bool) -> bool:
        if not is_duty(ev.priority) or not self.still_due(ev.kind, ev.payload):
            return False
        h = self.node.net.board.height
        retry = dict(ev.payload.get("retry") or {"since": h, "attempts": 0})
        retry["attempts"] += 1
        if provider and retry["attempts"] > MAX_PROVIDER_ATTEMPTS:
            self._drop(ev.kind, ev.payload, f"provider failed {MAX_PROVIDER_ATTEMPTS} times: {reason}")
            return False
        payload = {**ev.payload, "retry": retry}
        self.waiting.append(Waiting(ev.kind, payload, ev.priority,
                                    h + (PROVIDER_BACKOFF_BLOCKS if provider else 1), reason[:300]))
        self.node.net.log(self.node.id, "duty_deferred", kind=ev.kind, reason=reason[:300],
                          attempts=retry["attempts"], **_ref(payload))
        return True

    def _drop(self, kind: str, payload: dict[str, Any], reason: str) -> None:
        self.node.net.log(self.node.id, "duty_dropped", kind=kind, reason=reason[:300], **_ref(payload))


def _ref(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: payload[k] for k in ("ref", "bounty_id", "proposal_id") if k in payload}
