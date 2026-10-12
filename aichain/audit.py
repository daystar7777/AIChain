"""Hourly self-audit: deterministic checks of the board's books, independent of the LLM.

A finding means a bug or tampering, never a node's choice, so it is written to the journal
(event "audit_alert") and sent to the manager outbox (Telegram picks it up). Fixing is a human
job; the audit never changes state.
"""

from __future__ import annotations

from typing import Any

AUDIT_INTERVAL_BLOCKS = 720  # one hour at 5 s blocks


def run(board: Any) -> list[str]:
    problems: list[str] = []
    held = sum(a.balance_mP for a in board.accounts.values())
    if held + board.escrow_mP + board.burned_mP != board.issued_mP:
        problems.append(f"supply mismatch: balances {held} + escrow {board.escrow_mP} + burned "
                        f"{board.burned_mP} != issued {board.issued_mP} (mP)")
    if board.escrow_mP < 0:
        problems.append(f"negative escrow {board.escrow_mP}mP")
    for a in board.accounts.values():
        if a.balance_mP < 0:
            problems.append(f"{a.id}: negative balance {a.balance_mP}mP")
        if not 0 <= a.locked_mP <= max(a.balance_mP, 0):
            problems.append(f"{a.id}: locked {a.locked_mP}mP outside 0..balance {a.balance_mP}mP")
        if a.locked_mP != sum(x for x, _ in a.locks):
            problems.append(f"{a.id}: locked {a.locked_mP}mP != sum of locks {sum(x for x, _ in a.locks)}mP")
    in_escrow = sum(d.amount_mP for d in board.deposits.values() if d.state == "LOCKED")
    in_escrow += sum(b.reward_mP for b in board.bounties.values()
                     if str(b.state) in ("OPEN", "REVEAL", "JUDGING"))
    if in_escrow > board.escrow_mP:
        problems.append(f"escrow {board.escrow_mP}mP is less than open bounties and deposits {in_escrow}mP")
    return problems


def run_on_network(net: Any) -> list[str]:
    problems = run(net.board)
    net.log("core", "audit", height=net.board.height, ok=not problems, problems=problems[:20])
    if problems:
        net.log("core", "audit_alert", problems=problems[:20])
        net.manager_send("core", "⚠️ 자기 감사 이상 발견 (높이 %d):\n- %s" % (
            net.board.height, "\n- ".join(problems[:10])))
    return problems
