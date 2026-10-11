"""In-memory bounty board — the prototype stand-in for the chain (design doc 03, ch. 3).

No signatures, blocks or consensus: a single deterministic state machine driven by a
virtual block height. The rules (escrow, commit–reveal, bonds, settlement, burns,
settlement locks) follow 03-economy.md so the Mastermind sees the real protocol shape.
All amounts are integer mP.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable

from aichain.common import canonical_json, sha256_hex


class BoardError(Exception):
    """A request the protocol rejects. The message is shown to the Mastermind."""


class BountyState(StrEnum):
    OPEN = "OPEN"
    REVEAL = "REVEAL"
    JUDGING = "JUDGING"
    PAID = "PAID"
    AUTO_SPLIT = "AUTO_SPLIT"
    REFUNDED = "REFUNDED"
    EXPIRED = "EXPIRED"
    FINAL = "FINAL"


SETTLED_STATES = {BountyState.PAID, BountyState.AUTO_SPLIT, BountyState.REFUNDED}


@dataclass
class Params:
    min_reward_mP: int = 1000
    commit_blocks: int = 120
    reveal_blocks: int = 60
    judge_blocks: int = 240
    dispute_blocks: int = 720
    commit_bond_mP: int = 500
    max_answers: int = 5
    max_tags: int = 5
    burn_bp: int = 500  # 5% in basis points
    commit_blocks_range: tuple[int, int] = (12, 17280)


@dataclass
class Account:
    id: str
    balance_mP: int
    locked_mP: int = 0
    # (amount_mP, unlock_height)
    locks: list[tuple[int, int]] = field(default_factory=list)
    pubkey: str | None = None  # Ed25519 public key (hex) that signs this account's transactions

    @property
    def available_mP(self) -> int:
        return self.balance_mP - self.locked_mP


@dataclass
class Answer:
    answer_id: str
    bounty_id: str
    author: str
    commitment: str
    committed_at: int
    revealed: bool = False
    body_hash: str | None = None
    rejected_reason: str | None = None


@dataclass
class Bounty:
    bounty_id: str
    asker: str
    question_hash: str
    reward_mP: int
    tags: list[str]
    max_answers: int
    created_at: int
    commit_until: int
    reveal_until: int
    judge_until: int
    state: BountyState = BountyState.OPEN
    outcome: BountyState | None = None
    accepted: str | None = None
    settled_at: int | None = None
    answers: list[str] = field(default_factory=list)


@dataclass
class BoardEvent:
    """Something a node may need to react to. `to` is None for a broadcast."""

    kind: str
    to: str | None
    payload: dict[str, Any]


def transaction(method):
    """Mark a state-changing Board method: after it succeeds, its arguments are handed to the
    recorder (the chain log), so the state can be rebuilt by replaying the same calls."""
    sig = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.halted:
            raise BoardError(f"chain halted: {self.halted}")
        result = method(self, *args, **kwargs)
        if self.recorder is not None:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            body = {k: v for k, v in bound.arguments.items() if k != "self"}
            self.recorder(method.__name__, body)
        return result

    wrapper.is_transaction = True
    return wrapper


def commitment_of(body_hash: str, salt: str, author: str) -> str:
    return sha256_hex(f"{body_hash}{salt}{author}")


class Board:
    def __init__(self, params: Params | None = None) -> None:
        self.params = params or Params()
        self.height = 0
        self.accounts: dict[str, Account] = {}
        self.bounties: dict[str, Bounty] = {}
        self.answers: dict[str, Answer] = {}
        self.escrow_mP = 0
        self.burned_mP = 0
        self.issued_mP = 0
        self.rep_events: list[dict[str, Any]] = []
        self._seq = 0
        # Called as recorder(tx_type, body) after each successful transaction (see `transaction`).
        self.recorder: Callable[[str, dict[str, Any]], None] | None = None
        self.halted: str | None = None  # set when the chain log can no longer be written
        self._heartbeats: set[tuple[str, int]] = set()

    # ---- accounts ----

    @transaction
    def open_account(self, account_id: str, grant_mP: int, pubkey: str | None = None) -> Account:
        if account_id in self.accounts:
            raise BoardError(f"account {account_id} exists")
        acct = Account(account_id, grant_mP, pubkey=pubkey)
        self.accounts[account_id] = acct
        self.issued_mP += grant_mP
        return acct

    def apply(self, tx_type: str, body: dict[str, Any]) -> Any:
        """Re-apply a recorded transaction (replay)."""
        method = getattr(self, tx_type, None)
        if method is None or not getattr(method, "is_transaction", False):
            raise BoardError(f"unknown transaction type {tx_type}")
        return method(**body)

    def state_root(self) -> str:
        """Hash of the whole consensus state; equal on every honest replay (03/01 docs)."""
        state = {
            "height": self.height, "escrow": self.escrow_mP, "burned": self.burned_mP,
            "issued": self.issued_mP, "seq": self._seq,
            "accounts": {k: [a.balance_mP, a.locked_mP, sorted(a.locks), a.pubkey]
                         for k, a in sorted(self.accounts.items())},
            "bounties": {k: {**asdict(b), "state": str(b.state),
                             "outcome": str(b.outcome) if b.outcome else None}
                         for k, b in sorted(self.bounties.items())},
            "answers": {k: asdict(a) for k, a in sorted(self.answers.items())},
            "rep_events": sha256_hex(canonical_json(self.rep_events)),
        }
        return sha256_hex(canonical_json(state))

    def account(self, account_id: str) -> Account:
        try:
            return self.accounts[account_id]
        except KeyError:
            raise BoardError(f"unknown account {account_id}") from None

    def total_supply_check(self) -> bool:
        """issued == balances + escrow + burned (03 doc 6: supply = issued - burned)."""
        held = sum(a.balance_mP for a in self.accounts.values())
        return held + self.escrow_mP + self.burned_mP == self.issued_mP

    def _new_id(self, prefix: str, *parts: Any) -> str:
        self._seq += 1
        return prefix + sha256_hex("|".join(map(str, (self._seq, self.height, *parts))))[:12]

    def _debit_to_escrow(self, acct: Account, amount: int) -> None:
        if acct.available_mP < amount:
            raise BoardError(
                f"insufficient available balance: need {amount}mP, have {acct.available_mP}mP"
            )
        acct.balance_mP -= amount
        self.escrow_mP += amount

    def _pay_from_escrow(self, to: str, amount: int, *, lock: bool) -> None:
        if amount <= 0:
            return
        self.escrow_mP -= amount
        acct = self.account(to)
        acct.balance_mP += amount
        if lock:
            acct.locked_mP += amount
            acct.locks.append((amount, self.height + self.params.dispute_blocks))

    def _burn_from_escrow(self, amount: int) -> None:
        self.escrow_mP -= amount
        self.burned_mP += amount

    # ---- transactions ----

    @transaction
    def create_bounty(
        self,
        asker: str,
        question_hash: str,
        reward_mP: int,
        tags: list[str],
        max_answers: int | None = None,
        commit_blocks: int | None = None,
    ) -> Bounty:
        p = self.params
        acct = self.account(asker)
        if reward_mP < p.min_reward_mP:
            raise BoardError(f"reward must be at least {p.min_reward_mP}mP")
        if len(tags) > p.max_tags:
            raise BoardError(f"at most {p.max_tags} tags")
        max_answers = p.max_answers if max_answers is None else max_answers
        if not 1 <= max_answers <= 20:
            raise BoardError("max_answers must be between 1 and 20")
        commit_blocks = p.commit_blocks if commit_blocks is None else commit_blocks
        lo, hi = p.commit_blocks_range
        if not lo <= commit_blocks <= hi:
            raise BoardError(f"commit_blocks must be between {lo} and {hi}")
        self._debit_to_escrow(acct, reward_mP)
        commit_until = self.height + commit_blocks
        reveal_until = commit_until + p.reveal_blocks
        bounty = Bounty(
            bounty_id=self._new_id("b_", asker, question_hash),
            asker=asker,
            question_hash=question_hash,
            reward_mP=reward_mP,
            tags=list(tags),
            max_answers=max_answers,
            created_at=self.height,
            commit_until=commit_until,
            reveal_until=reveal_until,
            judge_until=reveal_until + p.judge_blocks,
        )
        self.bounties[bounty.bounty_id] = bounty
        return bounty

    def bounty(self, bounty_id: str) -> Bounty:
        try:
            return self.bounties[bounty_id]
        except KeyError:
            raise BoardError(f"unknown bounty {bounty_id}") from None

    @transaction
    def commit_answer(self, author: str, bounty_id: str, commitment: str) -> Answer:
        b = self.bounty(bounty_id)
        if b.state != BountyState.OPEN:
            raise BoardError(f"bounty is {b.state}, not OPEN")
        if author == b.asker:
            raise BoardError("cannot answer your own bounty")
        if any(self.answers[a].author == author for a in b.answers):
            raise BoardError("already answered this bounty")
        if len(b.answers) >= b.max_answers:
            raise BoardError("bounty has reached max_answers")
        self._debit_to_escrow(self.account(author), self.params.commit_bond_mP)
        ans = Answer(
            answer_id=self._new_id("a_", author, bounty_id, commitment),
            bounty_id=bounty_id,
            author=author,
            commitment=commitment,
            committed_at=self.height,
        )
        self.answers[ans.answer_id] = ans
        b.answers.append(ans.answer_id)
        return ans

    @transaction
    def reveal_answer(self, author: str, answer_id: str, body_hash: str, salt: str) -> None:
        ans = self.answers.get(answer_id)
        if ans is None or ans.author != author:
            raise BoardError("unknown answer")
        b = self.bounty(ans.bounty_id)
        if b.state != BountyState.REVEAL:
            raise BoardError(f"bounty is {b.state}, not REVEAL")
        if ans.revealed:
            raise BoardError("already revealed")
        if commitment_of(body_hash, salt, author) != ans.commitment:
            raise BoardError("reveal does not match commitment")
        ans.revealed = True
        ans.body_hash = body_hash
        # Revealing returns the commit bond.
        self._pay_from_escrow(author, self.params.commit_bond_mP, lock=False)

    def _judgeable(self, asker: str, bounty_id: str, answer_id: str) -> tuple[Bounty, Answer]:
        b = self.bounty(bounty_id)
        if b.asker != asker:
            raise BoardError("only the asker can judge")
        if b.state != BountyState.JUDGING:
            raise BoardError(f"bounty is {b.state}, not JUDGING")
        ans = self.answers.get(answer_id)
        if ans is None or ans.bounty_id != bounty_id or not ans.revealed:
            raise BoardError(f"{answer_id} is not a revealed answer of this bounty")
        return b, ans

    @transaction
    def reject_answer(self, asker: str, bounty_id: str, answer_id: str, reason: str) -> None:
        _, ans = self._judgeable(asker, bounty_id, answer_id)
        if ans.rejected_reason is not None:
            raise BoardError("already rejected")
        ans.rejected_reason = reason or "(no reason)"
        self._rep(ans.author, "answer_rejected", bounty_id=bounty_id)

    @transaction
    def accept_answer(self, asker: str, bounty_id: str, answer_id: str) -> list[BoardEvent]:
        b, ans = self._judgeable(asker, bounty_id, answer_id)
        if ans.rejected_reason is not None:
            raise BoardError("cannot accept a rejected answer")
        b.accepted = answer_id
        self._pay_from_escrow(ans.author, b.reward_mP, lock=True)
        self._rep(ans.author, "answer_accepted", bounty_id=bounty_id, value_mP=b.reward_mP)
        self._rep(asker, "judge_ok", bounty_id=bounty_id)
        return [self._settle(b, BountyState.PAID)]

    @transaction
    def heartbeat(self, account: str, epoch: int) -> None:
        """Liveness proof, one per account per epoch (720 blocks). Feeds reputation term U."""
        self.account(account)
        if epoch != self.height // 720:
            raise BoardError(f"heartbeat for epoch {epoch} at height {self.height}")
        if (account, epoch) in self._heartbeats:
            raise BoardError("already sent a heartbeat this epoch")
        self._heartbeats.add((account, epoch))
        self._rep(account, "heartbeat", epoch=epoch)

    def _rep(self, account: str, kind: str, **extra: Any) -> None:
        self.rep_events.append({"account": account, "kind": kind, "height": self.height, **extra})

    def _settle(self, b: Bounty, outcome: BountyState) -> BoardEvent:
        b.state = outcome
        b.outcome = outcome
        b.settled_at = self.height
        participants = sorted({b.asker, *(self.answers[a].author for a in b.answers)})
        return BoardEvent(
            "bounty_settled",
            None,
            {"bounty_id": b.bounty_id, "outcome": str(outcome), "participants": participants},
        )

    # ---- time ----

    def advance(self, blocks: int = 1) -> list[BoardEvent]:
        events: list[BoardEvent] = []
        for _ in range(blocks):
            if self.halted:
                break
            self.height += 1
            events.extend(self._on_block())
        return events

    def _on_block(self) -> list[BoardEvent]:
        h = self.height
        events: list[BoardEvent] = []
        for acct in self.accounts.values():
            if acct.locks:
                due = [lk for lk in acct.locks if lk[1] <= h]
                for lk in due:
                    acct.locks.remove(lk)
                    acct.locked_mP -= lk[0]
        for b in self.bounties.values():
            ev = self._transition(b, h)
            if ev:
                events.append(ev)
        return events

    def _transition(self, b: Bounty, h: int) -> BoardEvent | None:
        p = self.params
        if b.state == BountyState.OPEN and h >= b.commit_until:
            if not b.answers:
                self._pay_from_escrow(b.asker, b.reward_mP, lock=False)
                return self._settle_expired(b)
            b.state = BountyState.REVEAL
            return BoardEvent("reveal_open", None, {"bounty_id": b.bounty_id})
        if b.state == BountyState.REVEAL and h >= b.reveal_until:
            unrevealed = [a for a in b.answers if not self.answers[a].revealed]
            for _ in unrevealed:
                self._burn_from_escrow(p.commit_bond_mP)
            if len(unrevealed) == len(b.answers):
                self._pay_from_escrow(b.asker, b.reward_mP, lock=False)
                return self._settle_expired(b)
            b.state = BountyState.JUDGING
            return BoardEvent("judge_due", b.asker, {"bounty_id": b.bounty_id})
        if b.state == BountyState.JUDGING and h >= b.judge_until:
            winners = [
                self.answers[a].author
                for a in b.answers
                if self.answers[a].revealed and self.answers[a].rejected_reason is None
            ]
            burn = b.reward_mP * p.burn_bp // 10_000
            pool = b.reward_mP - burn
            if winners:
                share = pool // len(winners)
                for w in winners:
                    self._pay_from_escrow(w, share, lock=True)
                    self._rep(w, "answer_auto", bounty_id=b.bounty_id, value_mP=share)
                self._burn_from_escrow(b.reward_mP - share * len(winners))
                self._rep(b.asker, "judge_timeout", bounty_id=b.bounty_id)
                return self._settle(b, BountyState.AUTO_SPLIT)
            self._pay_from_escrow(b.asker, pool, lock=True)
            self._burn_from_escrow(burn)
            return self._settle(b, BountyState.REFUNDED)
        if b.state in SETTLED_STATES and b.settled_at is not None:
            if h >= b.settled_at + p.dispute_blocks:
                b.state = BountyState.FINAL
                return BoardEvent("bounty_final", None, {"bounty_id": b.bounty_id,
                                                         "outcome": str(b.outcome)})
        return None

    def _settle_expired(self, b: Bounty) -> BoardEvent:
        # EXPIRED refunds are not dispute-locked, so the bounty is terminal right away.
        return self._settle(b, BountyState.EXPIRED)
