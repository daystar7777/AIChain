"""Policy Guard: deterministic limits every write tool passes before reaching Core
(design doc 04, 3.4). The LLM expresses intent; this code decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BLOCKS_PER_HOUR = 720


class PolicyViolation(Exception):
    pass


@dataclass
class PolicyLimits:
    hourly_spend_ratio_bp: int = 2000  # 20% of balance
    hourly_spend_cap_mP: int = 200_000  # 200P
    bounty_ratio_bp: int = 1000  # 10% of balance per bounty
    writes_per_episode: int = 4
    pause_after_violations: int = 10


@dataclass
class PolicyGuard:
    node_id: str
    limits: PolicyLimits = field(default_factory=PolicyLimits)
    spends: list[tuple[int, int]] = field(default_factory=list)  # (height, amount_mP)
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
        if bounty and amount_mP > self.bounty_cap_mP(balance_mP):
            self._violate(
                f"bounty {amount_mP}mP exceeds per-bounty cap {self.bounty_cap_mP(balance_mP)}mP "
                f"(10% of balance)")
        limit = self.hourly_limit_mP(balance_mP)
        if self.spent_last_hour(height) + amount_mP > limit:
            self._violate(f"hourly spend limit {limit}mP would be exceeded")

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
