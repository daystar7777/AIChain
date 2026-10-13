"""Policy Guard: deterministic limits every write tool passes before reaching Core
(design doc 04, 3.4). The LLM expresses intent; this code decides.

A pause (10 consecutive violations, or an attempt to move more than half of the balance
within an hour) stops the Mastermind until the operator lifts it (`python -m aichain.admin
unpause <node>`); Core duties keep running. The node alerts the operator through `on_pause`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

BLOCKS_PER_HOUR = 720
BLOCKS_PER_DAY = 17_280


class PolicyViolation(Exception):
    pass


@dataclass
class PolicyLimits:
    hourly_spend_ratio_bp: int = 2000  # 20% of balance
    hourly_spend_cap_mP: int = 200_000  # 200P
    bounty_ratio_bp: int = 1000  # 10% of balance per bounty
    writes_per_episode: int = 4
    disputes_per_day: int = 5
    pause_after_violations: int = 10
    # Anomaly (04 doc 3.4, 07 doc 5): outflows within a rolling hour above 50% of the available
    # balance at the window start pause the Mastermind. Below the floor (dust) it never triggers.
    anomaly_outflow_bp: int = 5000
    anomaly_floor_mP: int = 10_000


@dataclass
class PolicyGuard:
    node_id: str
    limits: PolicyLimits = field(default_factory=PolicyLimits)
    spends: list[tuple[int, int]] = field(default_factory=list)  # (height, amount_mP)
    disputes: list[int] = field(default_factory=list)  # heights
    ratings: list[tuple[int, str]] = field(default_factory=list)  # (height, subject)
    writes_this_episode: int = 0
    consecutive_violations: int = 0
    paused: bool = False
    pause_reason: str | None = None
    # (height, available balance) each time it changed: the balance at the start of the hour.
    balance_marks: list[tuple[int, int]] = field(default_factory=list)
    on_pause: Callable[[str], None] | None = field(default=None, repr=False, compare=False)

    def start_episode(self) -> None:
        self.writes_this_episode = 0

    def hourly_limit_mP(self, balance_mP: int) -> int:
        return min(balance_mP * self.limits.hourly_spend_ratio_bp // 10_000,
                   self.limits.hourly_spend_cap_mP)

    def bounty_cap_mP(self, balance_mP: int) -> int:
        return balance_mP * self.limits.bounty_ratio_bp // 10_000

    def spent_last_hour(self, height: int) -> int:
        return sum(a for h, a in self.spends if h > height - BLOCKS_PER_HOUR)

    def check_write(self) -> None:
        if self.paused:
            raise PolicyViolation("mastermind is paused by Policy Guard; operator review required")
        if self.writes_this_episode >= self.limits.writes_per_episode:
            self._violate(f"write tool limit per episode ({self.limits.writes_per_episode}) reached")

    def observe_balance(self, height: int, balance_mP: int) -> None:
        """Core samples the available balance every block and before each spend; only changes
        are kept, back to the last one at or before the start of the hour."""
        if not self.balance_marks or self.balance_marks[-1][1] != balance_mP:
            self.balance_marks.append((height, balance_mP))
        start = height - BLOCKS_PER_HOUR
        while len(self.balance_marks) > 1 and self.balance_marks[1][0] <= start:
            self.balance_marks.pop(0)

    def balance_at_window_start(self, height: int) -> int:
        before = [b for h, b in self.balance_marks if h <= height - BLOCKS_PER_HOUR]
        # Nothing that old (node just started): the earliest balance seen.
        return before[-1] if before else self.balance_marks[0][1]

    def check_outflow(self, amount_mP: int, balance_mP: int, height: int) -> None:
        """Pause on an attempt to move more than half of what the node held an hour ago. The
        attempt counts, even if another limit would refuse it anyway."""
        self.observe_balance(height, balance_mP)
        moved = self.spent_last_hour(height) + amount_mP
        base = self.balance_at_window_start(height)
        if moved >= self.limits.anomaly_floor_mP and moved * 10_000 > base * self.limits.anomaly_outflow_bp:
            reason = (f"outflows of {moved}mP within an hour would exceed "
                      f"{self.limits.anomaly_outflow_bp / 100:g}% of the balance at the window start ({base}mP)")
            self.consecutive_violations += 1
            self.pause(reason)
            raise PolicyViolation(f"{reason}; mastermind paused, operator review required")

    def check_spend(self, amount_mP: int, balance_mP: int, height: int, *, bounty: bool) -> None:
        self.check_outflow(amount_mP, balance_mP, height)
        if bounty and amount_mP > self.bounty_cap_mP(balance_mP):
            self._violate(
                f"bounty {amount_mP}mP exceeds per-bounty cap {self.bounty_cap_mP(balance_mP)}mP "
                f"(10% of balance)")
        limit = self.hourly_limit_mP(balance_mP)
        if self.spent_last_hour(height) + amount_mP > limit:
            self._violate(f"hourly spend limit {limit}mP would be exceeded")

    def check_dispute(self, height: int) -> None:
        recent = [x for x in self.disputes if x > height - BLOCKS_PER_DAY]
        if len(recent) >= self.limits.disputes_per_day:
            self._violate(f"dispute limit {self.limits.disputes_per_day}/day reached")

    def check_rating(self, subject: str, height: int) -> None:
        if any(s == subject and h > height - BLOCKS_PER_DAY for h, s in self.ratings):
            self._violate(f"already rated {subject} today (1 per subject per day)")

    def record_spend(self, spend_mP: int, height: int) -> None:
        """Spending that is not a write call (e.g. per-use tool fees)."""
        if spend_mP:
            self.spends.append((height, spend_mP))

    def record_write(self, spend_mP: int = 0, height: int = 0) -> None:
        self.writes_this_episode += 1
        self.consecutive_violations = 0
        if spend_mP:
            self.spends.append((height, spend_mP))

    def _violate(self, reason: str) -> None:
        self.consecutive_violations += 1
        if self.consecutive_violations >= self.limits.pause_after_violations:
            self.pause(f"{self.consecutive_violations} consecutive violations, last: {reason}")
        raise PolicyViolation(reason)

    def pause(self, reason: str) -> None:
        if self.paused:
            return
        self.paused, self.pause_reason = True, reason
        if self.on_pause is not None:
            self.on_pause(reason)

    def unpause(self) -> None:
        """Operator decision only (admin API / CLI / Telegram)."""
        self.paused, self.pause_reason, self.consecutive_violations = False, None, 0

    # ---- persistence: spend/dispute/rating history is height-based, so it stays valid across restarts ----

    def dump(self) -> dict:
        return {"spends": self.spends[-500:], "disputes": self.disputes[-100:], "ratings": self.ratings[-200:],
                "consecutive_violations": self.consecutive_violations, "paused": self.paused,
                "pause_reason": self.pause_reason, "balance_marks": self.balance_marks[-200:]}

    def restore(self, d: dict) -> None:
        self.spends = [tuple(x) for x in d.get("spends", [])]
        self.disputes = list(d.get("disputes", []))
        self.ratings = [tuple(x) for x in d.get("ratings", [])]
        self.consecutive_violations = int(d.get("consecutive_violations", 0))
        self.paused = bool(d.get("paused", False))
        self.pause_reason = d.get("pause_reason")
        self.balance_marks = [tuple(x) for x in d.get("balance_marks", [])]
