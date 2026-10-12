"""Observation metrics (design docs 03 2.4, 04 5.4, 06 ch. 6, 08 M7).

Computed from the live network; the journal stores a time series so the viewer can show
how the economy and the nodes change.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from aichain.reputation import INCOME_KINDS


def gini(values: list[int]) -> float:
    """0 = perfectly equal, 1 = one account holds everything."""
    xs = sorted(v for v in values if v >= 0)
    n, total = len(xs), sum(xs)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2 * cum) / (n * total) - (n + 1) / n, 4)


def vote_agreement(proposals: list[Any]) -> dict[str, float]:
    """Pairwise share of proposals where two voters cast the same choice (06 doc 6)."""
    voters = sorted({v for p in proposals for v in p.votes})
    out: dict[str, float] = {}
    for a, b in combinations(voters, 2):
        both = [p for p in proposals if a in p.votes and b in p.votes]
        if both:
            out[f"{a}-{b}"] = round(sum(p.votes[a] == p.votes[b] for p in both) / len(both), 3)
    return out


def compute(net: Any) -> dict[str, Any]:
    board = net.board
    nodes = sorted(net.nodes)
    balances = {n: board.accounts[n].balance_mP for n in nodes}
    outcomes: dict[str, int] = {}
    for b in board.bounties.values():
        key = str(b.outcome or b.state)
        outcomes[key] = outcomes.get(key, 0) + 1
    income: dict[str, int] = {n: 0 for n in nodes}
    for e in board.rep_events:
        if e["account"] in income and e["kind"] in INCOME_KINDS:
            income[e["account"]] += e.get("value_mP", 0)
    circulating = sum(a.balance_mP for a in board.accounts.values())
    return {
        "height": board.height,
        "supply": {"issued_mP": board.issued_mP, "burned_mP": board.burned_mP,
                   "escrow_mP": board.escrow_mP, "circulating_mP": circulating},
        "gini": gini(list(balances.values())),
        "balances_mP": balances,
        "income_mP": income,
        "reputation": {n: round(net.reputation(n).value, 4) for n in nodes},
        "bounties": outcomes,
        "disputes": {s: sum(1 for d in board.disputes.values() if d.state == s)
                     for s in ("open", "upheld", "rejected", "void")},
        "ratings": len([r for r in board.ratings.values() if not r.void]),
        "tools": {"listed": len(board.tools), "licenses": len(board.licenses),
                  "requests_open": sum(1 for r in board.tool_requests.values() if r.state == "OPEN"),
                  "requests_fulfilled": sum(1 for r in board.tool_requests.values() if r.state == "FULFILLED")},
        "info_market": {"offers_open": sum(1 for o in board.offers.values() if o.state == "OPEN"),
                        "purchases": sum(1 for p in board.purchases.values() if p.state == "COMPLETED"),
                        "volume_mP": sum(p.price_mP for p in board.purchases.values() if p.state == "COMPLETED")},
        "loans": {st: sum(1 for x in board.loans.values() if x.state == st)
                  for st in ("REQUESTED", "ACTIVE", "REPAID", "DEFAULTED")},
        "savings_mP": sum(d.amount_mP for d in board.deposits.values() if d.state == "LOCKED"),
        "proposals": {s: sum(1 for p in board.proposals.values() if p.state == s)
                      for s in {p.state for p in board.proposals.values()}},
        "vote_agreement": vote_agreement(list(board.proposals.values())),
        "llm_calls_today": net.gateway.daily_usage(),
        "knowledge_notes": {n: len(net.nodes[n].memory.notes()) for n in nodes},
    }
