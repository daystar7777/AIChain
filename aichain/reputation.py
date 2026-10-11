"""Reputation computed deterministically from the board's reputation events (design doc 03, 4.3).

Integer fixed point (x 10^6) throughout, so every node gets the same value bit for bit:
time decay by exact halvings plus linear interpolation, and an integer log10.

    R = 0.30·A + 0.15·V + 0.15·J + 0.10·T + 0.10·W + 0.10·U + 0.10·S

Terms without events yet in this prototype (disputes, ratings) stay neutral at 0.5.
A new node starts at 0.45.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

E6 = 1_000_000
HALF_LIFE_BLOCKS = 120_960  # 7 days
EPOCH_BLOCKS = 720
WEIGHTS_E6 = {"A": 300_000, "V": 150_000, "J": 150_000, "T": 100_000, "W": 100_000,
              "U": 100_000, "S": 100_000}


def decay_e6(age_blocks: int) -> int:
    """0.5 ** (age / HALF_LIFE) in e6, integer only."""
    if age_blocks <= 0:
        return E6
    halvings, rest = divmod(age_blocks, HALF_LIFE_BLOCKS)
    if halvings >= 40:
        return 0
    base = E6 >> halvings
    # Linear between base and base/2 inside the current half-life (max error ~6%, deterministic).
    return base - (base // 2) * rest // HALF_LIFE_BLOCKS


def log10_e6(x_e6: int) -> int:
    """log10 of a positive e6 fixed-point number, in e6 (integer only, ~1e-3 accuracy)."""
    if x_e6 <= 0:
        raise ValueError("log10 of non-positive")
    result = 0
    while x_e6 >= 10 * E6:
        x_e6 //= 10
        result += E6
    while x_e6 < E6:
        x_e6 *= 10
        result -= E6
    # Now 1 <= x < 10: binary digits of log10 by repeated squaring.
    bit = E6 // 2
    for _ in range(20):
        x_e6 = x_e6 * x_e6 // E6
        if x_e6 >= 10 * E6:
            x_e6 //= 10
            result += bit
        bit //= 2
        if bit == 0:
            break
    return result


def _ratio_e6(num_e6: int, den_e6: int) -> int:
    return max(0, min(E6, num_e6 * E6 // den_e6)) if den_e6 > 0 else E6 // 2


@dataclass
class Reputation:
    account: str
    value_e6: int
    terms_e6: dict[str, int]

    @property
    def value(self) -> float:
        return self.value_e6 / E6


def compute(events: list[dict[str, Any]], account: str, height: int,
            bounties_created: int = 0) -> Reputation:
    w: dict[str, int] = {}  # decayed counts in e6
    income_mP = 0
    for e in events:
        if e["account"] != account:
            continue
        weight = decay_e6(height - e["height"])
        w[e["kind"]] = w.get(e["kind"], 0) + weight
        if e["kind"] in ("answer_accepted", "answer_auto", "tool_licensed", "tool_called"):
            income_mP += e.get("value_mP", 0)
    acc, auto = w.get("answer_accepted", 0), w.get("answer_auto", 0)
    rej, lost_ans = w.get("answer_rejected", 0), w.get("dispute_lost_answer", 0)
    terms = {
        "A": _ratio_e6(acc + auto // 2 + E6, acc + auto + rej + 3 * lost_ans + 2 * E6),
        "V": min(E6, log10_e6(E6 + income_mP * 1000) // 3),  # income in P: mP * 1000 = P in e6
        "J": E6 - _ratio_e6(w.get("judge_timeout", 0) + w.get("dispute_lost_rejection", 0),
                            bounties_created * E6 + 2 * E6),
        # T: tools sold (licenses), penalised by lost tool disputes; neutral without tools.
        "T": (_ratio_e6(E6 + w.get("tool_licensed", 0),
                        2 * E6 + w.get("tool_licensed", 0) + 3 * w.get("dispute_lost_tool", 0))
              if w.get("tool_licensed") else E6 // 2),
        "W": E6 // 2,
        "U": min(E6, w.get("heartbeat", 0) // EPOCH_BLOCKS),
        "S": E6 // 2,
    }
    value = sum(WEIGHTS_E6[k] * v for k, v in terms.items()) // E6
    return Reputation(account, value, terms)


def for_board(board: Any, account: str) -> Reputation:
    created = sum(1 for b in board.bounties.values() if b.asker == account)
    return compute(board.rep_events, account, board.height, created)
