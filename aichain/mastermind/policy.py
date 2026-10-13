"""Policy Guard: deterministic limits every write tool passes before reaching Core
(design doc 04, 3.4). The LLM expresses intent; this code decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

    def check_spend(self, amount_mP: int, balance_mP: int, height: int, *, bounty: bool) -> None:
        problem = self.spend_problem(amount_mP, balance_mP, height, bounty=bounty)
        if problem:
            self._violate(problem)

    def spend_problem(self, amount_mP: int, balance_mP: int, height: int, *, bounty: bool = False) -> str | None:
        """The limit this spend would break, or None. No side effects: Core uses it for spends
        the Mastermind does not make itself (e.g. automatic subscription renewals)."""
        if bounty and amount_mP > self.bounty_cap_mP(balance_mP):
            return (f"bounty {amount_mP}mP exceeds per-bounty cap {self.bounty_cap_mP(balance_mP)}mP "
                    f"(10% of balance)")
        limit = self.hourly_limit_mP(balance_mP)
        if self.spent_last_hour(height) + amount_mP > limit:
            return f"hourly spend limit {limit}mP would be exceeded"
        return None

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
            self.paused = True
        raise PolicyViolation(reason)

    # ---- persistence: spend/dispute/rating history is height-based, so it stays valid across restarts ----

    def dump(self) -> dict:
        return {"spends": self.spends[-500:], "disputes": self.disputes[-100:], "ratings": self.ratings[-200:],
                "consecutive_violations": self.consecutive_violations, "paused": self.paused}

    def restore(self, d: dict) -> None:
        self.spends = [tuple(x) for x in d.get("spends", [])]
        self.disputes = list(d.get("disputes", []))
        self.ratings = [tuple(x) for x in d.get("ratings", [])]
        self.consecutive_violations = int(d.get("consecutive_violations", 0))
        self.paused = bool(d.get("paused", False))
