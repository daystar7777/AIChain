"""In-memory bounty board — the prototype stand-in for the chain (design doc 03, ch. 3).

No signatures, blocks or consensus: a single deterministic state machine driven by a
virtual block height. The rules (escrow, commit–reveal, bonds, settlement, burns,
settlement locks) follow 03-economy.md so the Mastermind sees the real protocol shape.
All amounts are integer mP.
"""

from __future__ import annotations

import bisect
import functools
import inspect
import itertools
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable

from aichain.common import canonical_json, sha256_hex
from aichain.reputation import INCOME_KINDS
from aichain import reputation

# Consensus rules version. A chain starts at rules 1 (the behaviour every block before the first
# `upgrade_rules` transaction was made with) and moves forward only through that root-signed
# transaction, so replaying old blocks always applies the rules they were made under.
# 2: per-answer bonds, dispute holds, void deal disputes refund, loan default caps and request
#    expiry, delivered-purchase timeout, pairwise-netted contribution income, bounty options.
RULES_LATEST = 2
# Fields added after rules 1, with their defaults. They are left out of the state root while they
# hold the default, so a rules-1 state hashes exactly as it did before they existed.
LATE_FIELDS: dict[str, dict[str, Any]] = {
    "Answer": {"bond_mP": None},
    "Bounty": {"min_rep_e6": None},
    "Dispute": {"held_mP": 0},
    "Loan": {"request_until": None},
    "Purchase": {"confirm_until": None},
    "ToolListing": {"blocked": False},
    "Proposal": {"closed_hash": None},
}
INCOME_WINDOW_EPOCHS = 24  # rules 2: pairwise netting window for contribution income (one day)
LOAN_DEFAULT_PENALTY_BP = 1000  # rules 2: a defaulted lender recovers at most owed + 10%
LOAN_REQUEST_BLOCKS = 17_280  # rules 2: an unfunded loan request expires after one day
CONFIRM_BLOCKS = 720  # rules 2: a delivered purchase not confirmed within an hour completes


# What the block loop (`_on_block`) can still act on, per collection. Every object only moves
# forward through its states and never re-enters this set once it has left it, so the loop keeps
# an index of these objects instead of walking every object ever created (gap review B3).
LIVE_IN_BLOCK_LOOP: dict[str, Callable[[Any], bool]] = {
    "bounties": lambda b: b.state in (BountyState.OPEN, BountyState.REVEAL, BountyState.JUDGING)
    or (b.state in SETTLED_STATES and b.settled_at is not None),
    "offers": lambda o: o.state == "OPEN",
    "purchases": lambda p: p.state in ("PENDING", "DELIVERED"),
    "deposits": lambda d: d.state == "LOCKED",
    "loans": lambda loan: loan.state in ("REQUESTED", "ACTIVE"),
    "subscriptions": lambda s: s.state == "ACTIVE",
    "tool_requests": lambda r: r.state == "OPEN",
    "proposals": lambda p: p.state in ("VOTING", "TIMELOCK"),
    "disputes": lambda d: d.state == "open",
}
# Board attributes a snapshot carries, and the ones that belong to the running process.
SNAPSHOT_FIELDS = ("params", "rules", "height", "accounts", "bounties", "answers", "escrow_mP",
                   "burned_mP", "issued_mP", "rep_events", "_seq", "_heartbeats", "tools", "disputes",
                   "proposals", "tool_requests", "offers", "purchases", "subscriptions", "loans",
                   "deposits", "ratings", "licenses")
RUNTIME_FIELDS = {"recorder", "halted", "chain_state", "_on_block", "_live_index", "_live_seen"}


def _enc(obj: Any) -> dict[str, Any]:
    d = asdict(obj)
    for key, default in LATE_FIELDS.get(type(obj).__name__, {}).items():
        if d.get(key) == default:
            d.pop(key, None)
    return d


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
    dispute_resolve_blocks: int = 2880
    dispute_bond_bp: int = 1000  # 10% of the target amount, at least 1P
    rating_dispute_blocks: int = 720
    epoch_blocks: int = 720
    block_reward_mP: int = 1000
    reward_halving_blocks: int = 518_400
    issuance: bool = True
    vote_blocks: int = 2880
    timelock_blocks: int = 1440
    proposal_deposit_mP: int = 50_000
    operator_seat: bool = True
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
    bond_mP: int | None = None  # rules 2: the bond actually paid (rules 1 used the live parameter)


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
    # Locked settlement payments [account, amount_mP]: what an upheld dispute can claw back.
    payouts: list[list] = field(default_factory=list)
    min_rep_e6: int | None = None  # rules 2: answerers need at least this reputation


@dataclass
class Dispute:
    dispute_id: str
    opener: str
    target_kind: str  # answer | rejection | rating
    target_id: str
    claim_hash: str
    bond_mP: int
    defendant: str
    opened_height: int
    deadline: int
    bounty_id: str | None = None
    state: str = "open"  # open | upheld | rejected | void
    ruling_hash: str | None = None
    # rules 2: the defendant's funds moved into escrow while the dispute is open, so an upheld
    # ruling can still recover them after the settlement lock has expired.
    held_mP: int = 0


@dataclass
class Rating:
    rating_id: str
    rater: str
    subject: str
    score: int
    context_ref: str
    height: int
    void: bool = False


# Parameters a `param` proposal may change (06 doc 2). Anything else, including the root's
# veto and the voter set rules, is constitutional or needs a `seat` proposal.
GOVERNABLE_PARAMS = {
    "min_reward_mP": (1000, 100_000), "commit_blocks": (12, 17_280), "reveal_blocks": (12, 17_280),
    "judge_blocks": (12, 17_280), "dispute_blocks": (120, 17_280), "commit_bond_mP": (0, 10_000),
    "max_answers": (1, 20), "dispute_resolve_blocks": (720, 34_560), "dispute_bond_bp": (100, 5000),
    "block_reward_mP": (0, 10_000), "vote_blocks": (720, 17_280), "timelock_blocks": (360, 17_280),
    "proposal_deposit_mP": (1000, 500_000),
}
LLM_VOTERS = ("n1", "n2", "n3", "n4")
OPERATOR = "operator"


@dataclass
class Proposal:
    proposal_id: str
    proposer: str
    kind: str  # param | policy | upgrade | seat
    title: str
    spec_hash: str
    changes: list[dict[str, Any]]  # param: [{key, from, to}]; seat: [{key: "operator_seat", to}]
    deposit_mP: int
    created_height: int
    vote_until: int
    timelock_until: int | None = None
    state: str = "VOTING"  # VOTING TIMELOCK APPROVED ACTIVATED IMPLEMENTING REJECTED EXPIRED VETOED
    votes: dict[str, str] = field(default_factory=dict)  # voter -> yes | no | abstain
    tally: dict[str, int] | None = None
    closed_hash: str | None = None  # operator's note when an IMPLEMENTING proposal is closed


@dataclass
class ToolRequest:
    """A buy order for a tool (도구 구매 요청): the buyer locks a reward; the first submitted tool
    version that passes the buyer's acceptance tests (run by Core, recorded by root) wins it."""
    request_id: str
    buyer: str
    title: str
    spec_hash: str
    tests_hash: str
    reward_mP: int
    created_height: int
    deadline: int
    state: str = "OPEN"  # OPEN | FULFILLED | EXPIRED
    submissions: list[dict[str, Any]] = field(default_factory=list)
    winner: str | None = None
    winner_tool: list | None = None  # [tool_id, version]


# info: sealed content (many buyers). llm_calls / tool_runs: a lot of today's quota (one buyer).
OFFER_KINDS = {"info", "llm_calls", "tool_runs", "subscription", "storage"}


@dataclass
class Offer:
    """A sell offer in the deal market (팔자). kind "info": sealed content sold to many buyers,
    delivered end-to-end encrypted; `content_commit` = sha256 of the plaintext."""
    offer_id: str
    seller: str
    kind: str
    title: str
    terms_hash: str
    price_mP: int
    content_commit: str | None
    created_height: int
    expires: int
    state: str = "OPEN"  # OPEN | WITHDRAWN | EXPIRED | SOLD (resource lots sell once)
    sold: int = 0
    quantity: int = 1  # resource offers: calls or runs in the lot
    period_blocks: int = 0  # subscription: length of one paid period


@dataclass
class Purchase:
    purchase_id: str
    offer_id: str
    buyer: str
    seller: str
    price_mP: int
    height: int
    deliver_until: int
    state: str = "PENDING"  # PENDING DELIVERED COMPLETED REFUNDED DISPUTED
    ciphertext_hash: str | None = None
    released_height: int | None = None
    until: int | None = None  # storage leases: end of the lease
    confirm_until: int | None = None  # rules 2: a delivery not confirmed by then completes


@dataclass
class Subscription:
    """Recurring delivery (구독): one period is escrowed at a time; at the period's end the seller is
    paid if it delivered at least once in it, otherwise the period is refunded; then it renews."""
    sub_id: str
    offer_id: str
    subscriber: str
    seller: str
    price_mP: int
    period_blocks: int
    period_start: int
    state: str = "ACTIVE"  # ACTIVE | ENDED
    delivered_this_period: int = 0
    periods_paid: int = 0
    periods_refunded: int = 0
    deliveries: int = 0


@dataclass
class Loan:
    """A point loan (대여·차입). The borrower posts the request (collateral escrowed at once);
    a lender funds it; at `due` the chain repays automatically if the borrower can, otherwise the
    loan defaults: collateral goes to the lender and the borrower gets a loan_default event."""
    loan_id: str
    borrower: str
    principal_mP: int
    interest_mP: int
    collateral_mP: int
    term_blocks: int
    terms_hash: str
    created_height: int
    state: str = "REQUESTED"  # REQUESTED ACTIVE REPAID DEFAULTED CANCELLED
    lender: str | None = None
    due: int | None = None
    settled_height: int | None = None
    request_until: int | None = None  # rules 2: an unfunded request expires (collateral returned)


@dataclass
class Deposit:
    """Fixed-term savings (저축). Earns a share of each epoch's savings pool (10% of issuance) pro
    rata to the amount; paid out with the principal at maturity; early withdrawal forfeits it."""
    deposit_id: str
    owner: str
    amount_mP: int
    created_height: int
    unlock: int
    state: str = "LOCKED"  # LOCKED | MATURED | WITHDRAWN_EARLY
    earned_mP: int = 0


@dataclass
class ToolListing:
    tool_id: str
    version: int
    author: str
    code_hash: str
    price_install_mP: int
    price_call_mP: int
    published_height: int
    withdrawn: bool = False
    blocked: bool = False  # operator block: no new licenses and no further paid calls


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
    def __init__(self, params: Params | None = None, rules: int = RULES_LATEST) -> None:
        self.params = params or Params()
        # A fresh in-memory board uses the latest rules; the chain store resets this to 1 before
        # replaying a log and moves it forward with `upgrade_rules` (see chain/store.py attach).
        self.rules = rules
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
        self.tools: dict[tuple[str, int], ToolListing] = {}
        self.disputes: dict[str, Dispute] = {}
        self.proposals: dict[str, Proposal] = {}
        self.tool_requests: dict[str, ToolRequest] = {}
        self.offers: dict[str, Offer] = {}
        self.purchases: dict[str, Purchase] = {}
        self.subscriptions: dict[str, Subscription] = {}
        self.loans: dict[str, Loan] = {}
        self.deposits: dict[str, Deposit] = {}
        self.ratings: dict[str, Rating] = {}
        self.licenses: set[tuple[str, str, int]] = set()  # (account, tool_id, version)
        # Block-loop index (see LIVE_IN_BLOCK_LOOP): derived, rebuilt on demand, not consensus state.
        self._live_index: dict[str, dict[Any, None]] = {}
        self._live_seen: dict[str, int] = {}

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

    @transaction
    def upgrade_rules(self, version: int) -> None:
        """Root moves the chain to newer consensus rules from this height on (signed by root)."""
        if not self.rules < version <= RULES_LATEST:
            raise BoardError(f"rules {version} cannot follow rules {self.rules}")
        self.rules = version

    def state_root(self) -> str:
        """Hash of the whole consensus state; equal on every honest replay (03/01 docs)."""
        state = {
            "height": self.height, "escrow": self.escrow_mP, "burned": self.burned_mP,
            "issued": self.issued_mP, "seq": self._seq,
            "accounts": {k: [a.balance_mP, a.locked_mP, sorted(a.locks), a.pubkey]
                         for k, a in sorted(self.accounts.items())},
            "bounties": {k: {**_enc(b), "state": str(b.state),
                             "outcome": str(b.outcome) if b.outcome else None}
                         for k, b in sorted(self.bounties.items())},
            "answers": {k: _enc(a) for k, a in sorted(self.answers.items())},
            "rep_events": sha256_hex(canonical_json(self.rep_events)),
            "tools": {f"{k[0]}@{k[1]}": _enc(v) for k, v in sorted(self.tools.items())},
            "licenses": sorted(self.licenses),
            "disputes": {k: _enc(v) for k, v in sorted(self.disputes.items())},
            "ratings": {k: _enc(v) for k, v in sorted(self.ratings.items())},
            "proposals": {k: _enc(v) for k, v in sorted(self.proposals.items())},
            "params": asdict(self.params),
            "tool_requests": {k: _enc(v) for k, v in sorted(self.tool_requests.items())},
            "offers": {k: _enc(v) for k, v in sorted(self.offers.items())},
            "purchases": {k: _enc(v) for k, v in sorted(self.purchases.items())},
            "subscriptions": {k: _enc(v) for k, v in sorted(self.subscriptions.items())},
            "loans": {k: _enc(v) for k, v in sorted(self.loans.items())},
            "deposits": {k: _enc(v) for k, v in sorted(self.deposits.items())},
        }
        if self.rules > 1:
            state["rules"] = self.rules
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

    def escrow_obligations(self) -> int:
        """What escrow must hold right now, item by item; equal to `escrow_mP` when the books are
        right (used by the self-audit and the invariant tests)."""
        open_bounty = {BountyState.OPEN, BountyState.REVEAL, BountyState.JUDGING}
        total = sum(b.reward_mP for b in self.bounties.values() if b.state in open_bounty)
        total += sum(self._bond_of(a) for a in self.answers.values() if not a.revealed
                     and self.bounties[a.bounty_id].state in (BountyState.OPEN, BountyState.REVEAL))
        total += sum(d.bond_mP + d.held_mP for d in self.disputes.values() if d.state == "open")
        total += sum(p.price_mP for p in self.purchases.values()
                     if p.state in ("PENDING", "DELIVERED", "DISPUTED"))
        total += sum(x.price_mP for x in self.subscriptions.values() if x.state == "ACTIVE")
        total += sum(x.collateral_mP for x in self.loans.values() if x.state in ("REQUESTED", "ACTIVE"))
        total += sum(d.amount_mP + d.earned_mP for d in self.deposits.values() if d.state == "LOCKED")
        total += sum(r.reward_mP for r in self.tool_requests.values() if r.state == "OPEN")
        total += sum(p.deposit_mP for p in self.proposals.values() if p.state in ("VOTING", "TIMELOCK"))
        return total

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

    def _pay_from_escrow(self, to: str, amount: int, *, lock: bool, bounty: "Bounty | None" = None) -> None:
        if lock and bounty is not None and amount > 0:
            bounty.payouts.append([to, amount])
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
        reveal_blocks: int | None = None,
        judge_blocks: int | None = None,
        min_rep_e6: int | None = None,
    ) -> Bounty:
        """Rules 2 add per-bounty reveal/judge periods and a minimum answerer reputation
        (03 doc 3.1); older transactions never carry them, so they replay unchanged."""
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
        if (reveal_blocks, judge_blocks, min_rep_e6) != (None, None, None) and self.rules < 2:
            raise BoardError("reveal_blocks, judge_blocks and min_rep_e6 need rules 2")
        for name, v in (("reveal_blocks", reveal_blocks), ("judge_blocks", judge_blocks)):
            if v is not None and not 12 <= v <= 17_280:
                raise BoardError(f"{name} must be between 12 and 17280")
        reveal_blocks = p.reveal_blocks if reveal_blocks is None else reveal_blocks
        judge_blocks = p.judge_blocks if judge_blocks is None else judge_blocks
        if min_rep_e6 is not None and not 0 <= min_rep_e6 <= 900_000:
            raise BoardError("min_rep_e6 must be between 0 and 900000 (0.90)")
        self._debit_to_escrow(acct, reward_mP)
        commit_until = self.height + commit_blocks
        reveal_until = commit_until + reveal_blocks
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
            judge_until=reveal_until + judge_blocks,
            min_rep_e6=min_rep_e6,
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
        if b.min_rep_e6 is not None and reputation.for_board(self, author).value_e6 < b.min_rep_e6:
            raise BoardError(f"this bounty needs reputation of at least {b.min_rep_e6 / 1e6:.2f}")
        bond = self.params.commit_bond_mP
        self._debit_to_escrow(self.account(author), bond)
        ans = Answer(
            answer_id=self._new_id("a_", author, bounty_id, commitment),
            bounty_id=bounty_id,
            author=author,
            commitment=commitment,
            committed_at=self.height,
            bond_mP=bond if self.rules >= 2 else None,
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
        self._pay_from_escrow(author, self._bond_of(ans), lock=False)

    def _bond_of(self, ans: Answer) -> int:
        """Rules 1 read the live parameter, so a governance change in between minted or burned the
        difference; rules 2 refund or burn exactly what was paid."""
        return ans.bond_mP if ans.bond_mP is not None else self.params.commit_bond_mP

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
        self._pay_from_escrow(ans.author, b.reward_mP, lock=True, bounty=b)
        self._rep(ans.author, "answer_accepted", bounty_id=bounty_id, value_mP=b.reward_mP,
                  **self._payer(asker))
        self._rep(asker, "judge_ok", bounty_id=bounty_id)
        return [self._settle(b, BountyState.PAID)]

    # ---- disputes and ratings (03 doc 3.4, 4.2) ----

    def _take_back(self, account: str, amount: int) -> int:
        """Claw back up to `amount` from an account, preferring still-locked settlement funds."""
        acct = self.account(account)
        taken = min(amount, acct.balance_mP)
        acct.balance_mP -= taken
        locked_part = min(taken, acct.locked_mP)
        acct.locked_mP -= locked_part
        remaining = locked_part
        for lk in sorted(acct.locks, key=lambda x: x[1]):
            if remaining <= 0:
                break
            acct.locks.remove(lk)
            if lk[0] > remaining:
                acct.locks.append((lk[0] - remaining, lk[1]))
                remaining = 0
            else:
                remaining -= lk[0]
        return taken

    @transaction
    def open_dispute(self, opener: str, target_kind: str, target_id: str, claim_hash: str) -> Dispute:
        self.account(opener)
        p, h = self.params, self.height
        if target_kind == "answer":
            ans = self.answers.get(target_id)
            if ans is None:
                raise BoardError("unknown answer")
            b = self.bounty(ans.bounty_id)
            paid = sum(a for acct, a in b.payouts if acct == ans.author)
            if b.state not in (BountyState.PAID, BountyState.AUTO_SPLIT) or paid == 0:
                raise BoardError("only paid answers in their settlement lock period can be disputed")
            if opener in (b.asker, ans.author):
                raise BoardError("the asker and the author cannot open an answer dispute")
            amount, defendant, bounty_id = paid, ans.author, b.bounty_id
        elif target_kind == "rejection":
            ans = self.answers.get(target_id)
            if ans is None or ans.rejected_reason is None:
                raise BoardError("not a rejected answer")
            b = self.bounty(ans.bounty_id)
            if opener != ans.author:
                raise BoardError("only the rejected author can dispute a rejection")
            if b.state != BountyState.REFUNDED:
                raise BoardError("rejection disputes apply to refunded bounties in their lock period")
            revealed = sum(1 for a in b.answers if self.answers[a].revealed)
            amount, defendant, bounty_id = b.reward_mP // max(revealed, 1), b.asker, b.bounty_id
        elif target_kind == "tool":
            tool_id, _, ver = target_id.rpartition("@")
            listing = self.tools.get((tool_id, int(ver) if ver.isdigit() else -1))
            if listing is None:
                raise BoardError("target_id must be <tool_id>@<version> of a published tool")
            if (opener, tool_id, listing.version) not in self.licenses:
                raise BoardError("only license holders can dispute a tool")
            amount, defendant, bounty_id = listing.price_install_mP, listing.author, None
        elif target_kind == "deal":
            pu = self.purchases.get(target_id)
            if pu is None or pu.buyer != opener:
                raise BoardError("only the buyer can dispute a purchase")
            if pu.state != "COMPLETED" or h > (pu.released_height or 0) + p.dispute_blocks:
                raise BoardError("completed purchases can be disputed during their lock period")
            amount, defendant, bounty_id = pu.price_mP, pu.seller, None
        elif target_kind == "rating":
            r = self.ratings.get(target_id)
            if r is None or r.void:
                raise BoardError("unknown rating")
            if opener != r.subject:
                raise BoardError("only the rated account can dispute a rating")
            if h > r.height + p.rating_dispute_blocks:
                raise BoardError("rating dispute window has passed")
            amount, defendant, bounty_id = 0, r.rater, None
        else:
            raise BoardError("target_kind must be answer, rejection, rating, deal or tool")
        if any(d.target_id == target_id and d.state == "open" for d in self.disputes.values()):
            raise BoardError("this target already has an open dispute")
        bond = 1000 if target_kind == "rating" else max(1000, amount * p.dispute_bond_bp // 10_000)
        self._debit_to_escrow(self.account(opener), bond)
        d = Dispute(self._new_id("d_", opener, target_id), opener, target_kind, target_id, claim_hash,
                    bond, defendant, h, h + p.dispute_resolve_blocks, bounty_id)
        if self.rules >= 2 and amount > 0:
            # The settlement lock (dispute_blocks) is shorter than the ruling deadline, so hold the
            # disputed amount in escrow now; an upheld ruling then recovers it in full.
            d.held_mP = self._take_back(defendant, amount)
            self.escrow_mP += d.held_mP
        self.disputes[d.dispute_id] = d
        return d

    @transaction
    def resolve_dispute(self, resolver: str, dispute_id: str, ruling: str, ruling_hash: str) -> Dispute:
        """Root operator ruling (03 doc 3.4). The signature check makes resolver == root."""
        d = self.disputes.get(dispute_id)
        if d is None or d.state != "open":
            raise BoardError("no open dispute with that id")
        if ruling not in ("uphold", "reject", "void"):
            raise BoardError("ruling must be uphold, reject or void")
        self._settle_dispute(d, ruling, ruling_hash)
        return d

    def _settle_dispute(self, d: Dispute, ruling: str, ruling_hash: str | None) -> None:
        d.ruling_hash = ruling_hash
        if self.rules >= 2:
            self._settle_dispute_v2(d, ruling)
            return
        if ruling == "void":
            d.state = "void"
            self._pay_from_escrow(d.opener, d.bond_mP, lock=False)
            return
        if d.target_kind == "deal" and self.purchases[d.target_id].state == "DISPUTED":
            pu = self.purchases[d.target_id]  # automatic dispute: payment still in escrow
            if ruling == "uphold":
                d.state, pu.state = "upheld", "REFUNDED"
                self._pay_from_escrow(pu.buyer, pu.price_mP, lock=False)
                self._rep(pu.seller, "dispute_lost_deal", dispute_id=d.dispute_id)
            else:
                d.state = "rejected" if ruling == "reject" else "void"
                pu.state, pu.released_height = "COMPLETED", self.height
                self._pay_from_escrow(pu.seller, pu.price_mP, lock=True)
                self.offers[pu.offer_id].sold += 1
                self._rep(pu.seller, "info_sold", offer_id=pu.offer_id, value_mP=pu.price_mP)
            return
        if ruling == "reject":
            d.state = "rejected"
            half = d.bond_mP // 2
            self._burn_from_escrow(d.bond_mP - half)
            self._pay_from_escrow(d.defendant, half, lock=False)
            self._rep(d.opener, "dispute_lost", dispute_id=d.dispute_id)
            self._rep(d.defendant, "dispute_won_defense", dispute_id=d.dispute_id)
            return
        d.state = "upheld"
        self._pay_from_escrow(d.opener, d.bond_mP, lock=False)
        self._rep(d.opener, "dispute_won", dispute_id=d.dispute_id)
        if d.target_kind == "answer":
            b = self.bounty(d.bounty_id)
            paid = sum(a for acct, a in b.payouts if acct == d.defendant)
            taken = self._take_back(d.defendant, paid)
            finder = taken // 5
            self.account(d.opener).balance_mP += finder
            self.account(b.asker).balance_mP += taken - finder
            b.payouts = [x for x in b.payouts if x[0] != d.defendant]
            self._rep(d.defendant, "dispute_lost_answer", dispute_id=d.dispute_id)
        elif d.target_kind == "rejection":
            b = self.bounty(d.bounty_id)
            revealed = sum(1 for a in b.answers if self.answers[a].revealed)
            share = b.reward_mP // max(revealed, 1)
            taken = self._take_back(b.asker, share)
            self.account(d.opener).balance_mP += taken
            self._rep(b.asker, "dispute_lost_rejection", dispute_id=d.dispute_id)
            for e in self.rep_events:  # the rejection no longer counts against the author
                if e["kind"] == "answer_rejected" and e["account"] == d.opener and e.get("bounty_id") == b.bounty_id:
                    e["kind"] = "answer_rejected_overturned"
        elif d.target_kind == "tool":
            tool_id, _, ver = d.target_id.rpartition("@")
            listing = self.tools[(tool_id, int(ver))]
            taken = self._take_back(listing.author, listing.price_install_mP)
            self.account(d.opener).balance_mP += taken
            listing.withdrawn = True
            self._rep(listing.author, "dispute_lost_tool", dispute_id=d.dispute_id)
        elif d.target_kind == "deal":
            pu = self.purchases[d.target_id]
            taken = self._take_back(pu.seller, pu.price_mP)
            self.account(pu.buyer).balance_mP += taken
            pu.state = "REFUNDED"
            self._rep(pu.seller, "dispute_lost_deal", dispute_id=d.dispute_id)
        elif d.target_kind == "rating":
            self.ratings[d.target_id].void = True
            for e in self.rep_events:
                if e["kind"] == "rated" and e.get("rating_id") == d.target_id:
                    e["kind"] = "rated_void"

    def _settle_dispute_v2(self, d: Dispute, ruling: str) -> None:
        if d.target_kind == "deal" and self.purchases[d.target_id].state == "DISPUTED":
            pu = self.purchases[d.target_id]  # automatic dispute: payment still in escrow, bond 0
            self._pay_from_escrow(d.opener, d.bond_mP, lock=False)
            if ruling == "reject":
                d.state = "rejected"
                self._complete_purchase(pu)
                return
            # Upheld, or nobody ruled in time: the buyer never got content matching the commitment.
            d.state = "upheld" if ruling == "uphold" else "void"
            pu.state = "REFUNDED"
            self._pay_from_escrow(pu.buyer, pu.price_mP, lock=False)
            if ruling == "uphold":
                self._rep(pu.seller, "dispute_lost_deal", dispute_id=d.dispute_id)
            return
        if ruling in ("void", "reject"):
            self._pay_from_escrow(d.defendant, d.held_mP, lock=False)  # the hold goes back
            if ruling == "void":
                d.state = "void"
                self._pay_from_escrow(d.opener, d.bond_mP, lock=False)
                return
            d.state = "rejected"
            half = d.bond_mP // 2
            self._burn_from_escrow(d.bond_mP - half)
            self._pay_from_escrow(d.defendant, half, lock=False)
            self._rep(d.opener, "dispute_lost", dispute_id=d.dispute_id)
            self._rep(d.defendant, "dispute_won_defense", dispute_id=d.dispute_id)
            return
        d.state = "upheld"
        self._pay_from_escrow(d.opener, d.bond_mP, lock=False)
        self._rep(d.opener, "dispute_won", dispute_id=d.dispute_id)

        def recover(amount: int) -> int:
            """The held amount plus whatever more the defendant still has, now in escrow."""
            extra = self._take_back(d.defendant, max(0, amount - d.held_mP))
            self.escrow_mP += extra
            return d.held_mP + extra

        if d.target_kind == "answer":
            b = self.bounty(d.bounty_id)
            taken = recover(sum(a for acct, a in b.payouts if acct == d.defendant))
            finder = taken // 5
            self._pay_from_escrow(d.opener, finder, lock=False)
            self._pay_from_escrow(b.asker, taken - finder, lock=False)
            b.payouts = [x for x in b.payouts if x[0] != d.defendant]
            self._rep(d.defendant, "dispute_lost_answer", dispute_id=d.dispute_id)
        elif d.target_kind == "rejection":
            b = self.bounty(d.bounty_id)
            revealed = sum(1 for a in b.answers if self.answers[a].revealed)
            self._pay_from_escrow(d.opener, recover(b.reward_mP // max(revealed, 1)), lock=False)
            self._rep(b.asker, "dispute_lost_rejection", dispute_id=d.dispute_id)
            for e in self.rep_events:  # the rejection no longer counts against the author
                if e["kind"] == "answer_rejected" and e["account"] == d.opener and e.get("bounty_id") == b.bounty_id:
                    e["kind"] = "answer_rejected_overturned"
        elif d.target_kind == "tool":
            tool_id, _, ver = d.target_id.rpartition("@")
            listing = self.tools[(tool_id, int(ver))]
            self._pay_from_escrow(d.opener, recover(listing.price_install_mP), lock=False)
            listing.withdrawn = True
            self._rep(listing.author, "dispute_lost_tool", dispute_id=d.dispute_id)
        elif d.target_kind == "deal":
            pu = self.purchases[d.target_id]
            self._pay_from_escrow(pu.buyer, recover(pu.price_mP), lock=False)
            pu.state = "REFUNDED"
            self._rep(pu.seller, "dispute_lost_deal", dispute_id=d.dispute_id)
        elif d.target_kind == "rating":
            self.ratings[d.target_id].void = True
            for e in self.rep_events:
                if e["kind"] == "rated" and e.get("rating_id") == d.target_id:
                    e["kind"] = "rated_void"

    def _rating_context(self, ref: str) -> tuple[set[str], bool]:
        """Who took part in a deal and whether it is settled: bounty, purchase, loan, tool request,
        subscription."""
        if ref in self.bounties:
            b = self.bounties[ref]
            return ({b.asker, *(self.answers[a].author for a in b.answers)},
                    b.state in SETTLED_STATES | {BountyState.FINAL, BountyState.EXPIRED})
        if ref in self.purchases:
            p = self.purchases[ref]
            return {p.buyer, p.seller}, p.state in ("COMPLETED", "REFUNDED")
        if ref in self.loans:
            l = self.loans[ref]
            return {l.borrower, *( [l.lender] if l.lender else [])}, l.state in ("REPAID", "DEFAULTED")
        if ref in self.tool_requests:
            r = self.tool_requests[ref]
            return {r.buyer, *(x["seller"] for x in r.submissions)}, r.state in ("FULFILLED", "EXPIRED")
        if ref in self.subscriptions:
            x = self.subscriptions[ref]
            return {x.subscriber, x.seller}, x.periods_paid + x.periods_refunded > 0 or x.state == "ENDED"
        raise BoardError(f"unknown deal {ref}")

    @transaction
    def rate(self, rater: str, subject: str, score: int, context_ref: str) -> Rating:
        """-2..+2 about someone you dealt with in one bounty (01 doc 3.2), once per context."""
        if rater == subject:
            raise BoardError("cannot rate yourself")
        if score not in (-2, -1, 0, 1, 2):
            raise BoardError("score must be -2..2")
        involved, settled = self._rating_context(context_ref)
        if rater not in involved or subject not in involved:
            raise BoardError("both accounts must have taken part in this deal")
        if not settled:
            raise BoardError("rate after the deal is settled")
        if any(r.rater == rater and r.subject == subject and r.context_ref == context_ref
               for r in self.ratings.values()):
            raise BoardError("already rated for this bounty")
        r = Rating(self._new_id("r_", rater, subject, context_ref), rater, subject, score,
                   context_ref, self.height)
        self.ratings[r.rating_id] = r
        self._rep(subject, "rated", rating_id=r.rating_id, rater=rater, score=score)
        return r

    # ---- governance (06 doc) ----

    def voters(self) -> list[str]:
        seats = [v for v in LLM_VOTERS if v in self.accounts]
        if self.params.operator_seat and OPERATOR in self.accounts:
            seats.append(OPERATOR)
        return seats

    @transaction
    def create_proposal(self, proposer: str, kind: str, title: str, spec_hash: str,
                        changes: list[dict[str, Any]]) -> Proposal:
        p, h = self.params, self.height
        if proposer not in self.voters():
            raise BoardError("only voting accounts can propose")
        if any(x.proposer == proposer and x.state in ("VOTING", "TIMELOCK") for x in self.proposals.values()):
            raise BoardError("one open proposal per account")
        if kind == "param":
            if not changes:
                raise BoardError("a param proposal needs changes")
            resolved = []
            for c in changes:
                key, to = c.get("key"), c.get("to")
                if key not in GOVERNABLE_PARAMS:
                    raise BoardError(f"{key} is not a governable parameter (constitution or other kind)")
                lo, hi = GOVERNABLE_PARAMS[key]
                if not isinstance(to, int) or not lo <= to <= hi:
                    raise BoardError(f"{key} must be an integer in [{lo}, {hi}]")
                resolved.append({"key": key, "from": getattr(p, key), "to": to})
            changes = resolved
        elif kind == "seat":
            if [c.get("key") for c in changes] != ["operator_seat"] or not isinstance(changes[0].get("to"), bool):
                raise BoardError("seat proposals change only operator_seat (true/false)")
            changes = [{"key": "operator_seat", "from": p.operator_seat, "to": changes[0]["to"]}]
        elif kind in ("policy", "upgrade"):
            changes = []
        else:
            raise BoardError("kind must be param, policy, upgrade or seat")
        self._debit_to_escrow(self.account(proposer), p.proposal_deposit_mP)
        prop = Proposal(self._new_id("p_", proposer, title), proposer, kind, title[:200], spec_hash,
                        changes, p.proposal_deposit_mP, h, h + p.vote_blocks)
        self.proposals[prop.proposal_id] = prop
        return prop

    @transaction
    def vote(self, voter: str, proposal_id: str, choice: str, reason_hash: str) -> None:
        prop = self.proposals.get(proposal_id)
        if prop is None or prop.state != "VOTING":
            raise BoardError("no proposal open for voting with that id")
        if voter not in self.voters():
            raise BoardError(f"{voter} has no vote")
        if choice not in ("yes", "no", "abstain"):
            raise BoardError("choice must be yes, no or abstain")
        if not reason_hash:
            raise BoardError("a vote needs a reason")
        prop.votes[voter] = choice  # the last vote counts

    @transaction
    def veto_proposal(self, vetoer: str, proposal_id: str, reason_hash: str) -> None:
        """Root veto during the timelock (constitutional; 06 doc 3.3). Signed by root."""
        prop = self.proposals.get(proposal_id)
        if prop is None or prop.state != "TIMELOCK":
            raise BoardError("only proposals in their timelock can be vetoed")
        prop.state = "VETOED"
        self._burn_from_escrow(prop.deposit_mP)

    def _tally(self, prop: Proposal) -> BoardEvent:
        votes = list(prop.votes.values())
        yes, no, abstain = votes.count("yes"), votes.count("no"), votes.count("abstain")
        prop.tally = {"yes": yes, "no": no, "abstain": abstain, "seats": len(self.voters())}
        cast = yes + no + abstain
        if prop.kind == "seat":
            quorum, passed = 4, (yes + no) > 0 and yes * 3 >= 2 * (yes + no)
        elif prop.kind == "upgrade":
            quorum, passed = 3, yes > no and yes >= 3
        else:
            quorum, passed = 3, yes > no
        if cast < quorum:
            prop.state = "EXPIRED"
            self._pay_from_escrow(prop.proposer, prop.deposit_mP, lock=False)
        elif not passed:
            prop.state = "REJECTED"
            if (yes + no) and no * 3 >= 2 * (yes + no):
                self._burn_from_escrow(prop.deposit_mP)
            else:
                self._pay_from_escrow(prop.proposer, prop.deposit_mP, lock=False)
        else:
            prop.state = "TIMELOCK"
            prop.timelock_until = self.height + self.params.timelock_blocks
        return BoardEvent("proposal_tallied", None, {"proposal_id": prop.proposal_id,
                                                     "state": prop.state, "tally": prop.tally})

    def _approve(self, prop: Proposal) -> BoardEvent:
        self._pay_from_escrow(prop.proposer, prop.deposit_mP, lock=False)
        if prop.kind in ("param", "seat"):
            for c in prop.changes:
                setattr(self.params, c["key"], c["to"])
            prop.state = "ACTIVATED"
        else:
            prop.state = "IMPLEMENTING"  # people implement and release (06 doc 4)
        return BoardEvent("proposal_approved", None, {"proposal_id": prop.proposal_id,
                                                      "state": prop.state, "changes": prop.changes})

    @transaction
    def close_proposal(self, closer: str, proposal_id: str, outcome: str, note_hash: str) -> None:
        """Root closes an approved policy/upgrade proposal once people implemented (or dropped) it
        (06 doc 4). Signed by root."""
        prop = self.proposals.get(proposal_id)
        if prop is None or prop.state != "IMPLEMENTING":
            raise BoardError("only IMPLEMENTING proposals can be closed")
        if outcome not in ("implemented", "abandoned"):
            raise BoardError("outcome must be implemented or abandoned")
        prop.state = "IMPLEMENTED" if outcome == "implemented" else "ABANDONED"
        prop.closed_hash = note_hash

    # ---- tool market (05 doc ch. 2–3, 01 doc 3.2) ----

    @transaction
    def block_tool(self, blocker: str, tool_id: str, version: int, reason_hash: str) -> None:
        """Operator block (05 doc 4.3), signed by root: withdrawn, no new licenses, no paid calls."""
        listing = self._listing(tool_id, version)
        if listing.blocked:
            raise BoardError("tool is already blocked")
        listing.blocked = listing.withdrawn = True
        self._rep(listing.author, "tool_blocked", tool_id=tool_id, version=version, reason_hash=reason_hash)

    @transaction
    def publish_tool(self, author: str, tool_id: str, version: int, code_hash: str,
                     price_install_mP: int, price_call_mP: int) -> ToolListing:
        self.account(author)
        if not tool_id.startswith(f"{author}."):
            raise BoardError(f"tool id must start with '{author}.'")
        if (tool_id, version) in self.tools:
            raise BoardError(f"{tool_id} v{version} is already published")
        if not 0 <= price_install_mP <= 100_000 or not 0 <= price_call_mP <= 10_000:
            raise BoardError("prices out of range (install <= 100P, call <= 10P)")
        listing = ToolListing(tool_id, version, author, code_hash, price_install_mP, price_call_mP,
                              self.height)
        self.tools[(tool_id, version)] = listing
        return listing

    @transaction
    def withdraw_tool(self, author: str, tool_id: str, version: int) -> None:
        listing = self._listing(tool_id, version)
        if listing.author != author:
            raise BoardError("only the author can withdraw a tool")
        listing.withdrawn = True  # existing licenses keep working

    def _listing(self, tool_id: str, version: int) -> ToolListing:
        listing = self.tools.get((tool_id, version))
        if listing is None:
            raise BoardError(f"{tool_id} v{version} is not published")
        return listing

    def _transfer(self, payer: str, payee: str, amount: int) -> None:
        src, dst = self.account(payer), self.account(payee)
        if src.available_mP < amount:
            raise BoardError(f"insufficient available balance: need {amount}mP, have {src.available_mP}mP")
        src.balance_mP -= amount
        dst.balance_mP += amount

    @transaction
    def license_tool(self, buyer: str, tool_id: str, version: int) -> None:
        listing = self._listing(tool_id, version)
        if listing.withdrawn:
            raise BoardError("tool is withdrawn")
        if buyer == listing.author:
            raise BoardError("cannot buy your own tool")
        if (buyer, tool_id, version) in self.licenses:
            raise BoardError("already licensed")
        self._transfer(buyer, listing.author, listing.price_install_mP)
        self.licenses.add((buyer, tool_id, version))
        self._rep(listing.author, "tool_licensed", tool_id=tool_id, version=version,
                  value_mP=listing.price_install_mP, **self._payer(buyer))

    @transaction
    def pay_tool_use(self, user: str, tool_id: str, version: int, calls: int) -> None:
        listing = self._listing(tool_id, version)
        if (user, tool_id, version) not in self.licenses:
            raise BoardError("no license for this tool version")
        if listing.blocked:
            raise BoardError("this tool version was blocked by the operator")
        if not 1 <= calls <= 100:
            raise BoardError("calls must be 1..100")
        amount = calls * listing.price_call_mP
        self._transfer(user, listing.author, amount)
        self._rep(listing.author, "tool_called", tool_id=tool_id, version=version,
                  calls=calls, value_mP=amount, **self._payer(user))

    # ---- deal market: sell offers and purchases ----

    @transaction
    def create_offer(self, seller: str, kind: str, title: str, terms_hash: str, price_mP: int,
                     content_commit: str | None, expires_blocks: int, quantity: int = 1,
                     period_blocks: int = 0) -> Offer:
        self.account(seller)
        if kind not in OFFER_KINDS:
            raise BoardError(f"offer kind must be one of {sorted(OFFER_KINDS)}")
        if kind == "info" and not content_commit:
            raise BoardError("an info offer needs a content commitment")
        if kind in ("llm_calls", "tool_runs") and not 1 <= quantity <= 100:
            raise BoardError("resource lots hold 1..100 units")
        if kind == "subscription" and not 720 <= period_blocks <= 17_280 * 7:
            raise BoardError("a subscription period is between 1 hour and 7 days")
        if kind == "storage" and (not 1 <= quantity <= 10_240 or not 720 <= period_blocks <= 17_280 * 30):
            raise BoardError("storage leases are 1..10240 KB for 1 hour..30 days")
        if not 0 < price_mP <= 100_000:
            raise BoardError("price must be 1mP..100P")
        if not 720 <= expires_blocks <= 17_280 * 30:
            raise BoardError("offers last between 1 hour and 30 days")
        if sum(1 for o in self.offers.values() if o.seller == seller and o.state == "OPEN") >= 20:
            raise BoardError("at most 20 open offers per account")
        o = Offer(self._new_id("o_", seller, title), seller, kind, title[:200], terms_hash, price_mP,
                  content_commit, self.height, self.height + expires_blocks,
                  quantity=quantity if kind in ("llm_calls", "tool_runs", "storage") else 1,
                  period_blocks=period_blocks if kind in ("subscription", "storage") else 0)
        self.offers[o.offer_id] = o
        return o

    @transaction
    def withdraw_offer(self, seller: str, offer_id: str) -> None:
        o = self.offers.get(offer_id)
        if o is None or o.seller != seller or o.state != "OPEN":
            raise BoardError("no open offer of yours with that id")
        o.state = "WITHDRAWN"

    @transaction
    def buy_offer(self, buyer: str, offer_id: str) -> Purchase:
        o = self.offers.get(offer_id)
        if o is None or o.state != "OPEN":
            raise BoardError("no open offer with that id")
        if buyer == o.seller:
            raise BoardError("cannot buy your own offer")
        if o.kind == "subscription":
            raise BoardError("use subscribe for subscription offers")
        if any(p.offer_id == offer_id and p.buyer == buyer and p.state not in ("REFUNDED",)
               for p in self.purchases.values()):
            raise BoardError("already bought")
        if o.kind != "info" and any(p.offer_id == offer_id and p.state in ("PENDING", "DELIVERED")
                                    for p in self.purchases.values()):
            raise BoardError("this resource lot is being sold to someone else")
        self._debit_to_escrow(self.account(buyer), o.price_mP)
        p = Purchase(self._new_id("pu_", buyer, offer_id), offer_id, buyer, o.seller, o.price_mP,
                     self.height, self.height + 240)
        self.purchases[p.purchase_id] = p
        return p

    @transaction
    def deliver_offer(self, seller: str, purchase_id: str, ciphertext_hash: str) -> None:
        p = self.purchases.get(purchase_id)
        if p is None or p.seller != seller or p.state != "PENDING":
            raise BoardError("no pending purchase to deliver")
        p.state, p.ciphertext_hash = "DELIVERED", ciphertext_hash
        if self.rules >= 2:
            p.confirm_until = self.height + CONFIRM_BLOCKS

    @transaction
    def confirm_delivery(self, buyer: str, purchase_id: str, ok: bool, reason_hash: str) -> Purchase:
        """Buyer's Core decrypted the delivery and checked it against the commitment."""
        p = self.purchases.get(purchase_id)
        if p is None or p.buyer != buyer or p.state != "DELIVERED":
            raise BoardError("no delivered purchase to confirm")
        if ok:
            self._complete_purchase(p)
        else:  # content does not match the commitment: the operator decides, payment stays escrowed
            p.state = "DISPUTED"
            d = Dispute(self._new_id("d_", buyer, purchase_id), buyer, "deal", purchase_id, reason_hash, 0,
                        p.seller, self.height, self.height + self.params.dispute_resolve_blocks)
            self.disputes[d.dispute_id] = d
        return p

    def _complete_purchase(self, p: Purchase) -> None:
        p.state, p.released_height = "COMPLETED", self.height
        self._pay_from_escrow(p.seller, p.price_mP, lock=True)
        offer = self.offers[p.offer_id]
        offer.sold += 1
        if offer.kind != "info":
            offer.state = "SOLD"
        if offer.kind == "storage":
            p.until = self.height + offer.period_blocks
        self._rep(p.seller, "info_sold", offer_id=p.offer_id, value_mP=p.price_mP, offer_kind=offer.kind,
                  **self._payer(p.buyer))

    # ---- subscriptions ----

    @transaction
    def subscribe(self, subscriber: str, offer_id: str) -> Subscription:
        o = self.offers.get(offer_id)
        if o is None or o.state != "OPEN" or o.kind != "subscription":
            raise BoardError("no open subscription offer with that id")
        if subscriber == o.seller:
            raise BoardError("cannot subscribe to yourself")
        if any(x.offer_id == offer_id and x.subscriber == subscriber and x.state == "ACTIVE"
               for x in self.subscriptions.values()):
            raise BoardError("already subscribed")
        self._debit_to_escrow(self.account(subscriber), o.price_mP)
        sub = Subscription(self._new_id("sb_", subscriber, offer_id), offer_id, subscriber, o.seller,
                           o.price_mP, o.period_blocks, self.height)
        self.subscriptions[sub.sub_id] = sub
        return sub

    @transaction
    def sub_deliver(self, seller: str, sub_id: str, ciphertext_hash: str) -> None:
        sub = self.subscriptions.get(sub_id)
        if sub is None or sub.seller != seller or sub.state != "ACTIVE":
            raise BoardError("no active subscription of yours with that id")
        sub.delivered_this_period += 1
        sub.deliveries += 1

    @transaction
    def cancel_subscription(self, subscriber: str, sub_id: str) -> Subscription:
        sub = self.subscriptions.get(sub_id)
        if sub is None or sub.subscriber != subscriber or sub.state != "ACTIVE":
            raise BoardError("no active subscription of yours with that id")
        self._close_period(sub)
        sub.state = "ENDED"
        return sub

    def _close_period(self, sub: Subscription) -> None:
        if sub.delivered_this_period > 0:
            self._pay_from_escrow(sub.seller, sub.price_mP, lock=True)
            sub.periods_paid += 1
            self._rep(sub.seller, "info_sold", offer_id=sub.offer_id, value_mP=sub.price_mP,
                      offer_kind="subscription", **self._payer(sub.subscriber))
        else:
            self._pay_from_escrow(sub.subscriber, sub.price_mP, lock=False)
            sub.periods_refunded += 1
        sub.delivered_this_period = 0

    # ---- loans ----

    @transaction
    def create_loan_request(self, borrower: str, principal_mP: int, interest_mP: int, collateral_mP: int,
                            term_blocks: int, terms_hash: str) -> Loan:
        self.account(borrower)
        if not 1000 <= principal_mP <= 200_000:
            raise BoardError("principal must be 1P..200P")
        if not 0 <= interest_mP <= principal_mP * 3 // 10:
            raise BoardError("interest must be 0..30% of the principal")
        if not 720 <= term_blocks <= 17_280 * 7:
            raise BoardError("term must be 1 hour..7 days")
        if collateral_mP < 0:
            raise BoardError("collateral cannot be negative")
        if sum(1 for x in self.loans.values() if x.borrower == borrower and x.state in ("REQUESTED", "ACTIVE")) >= 3:
            raise BoardError("at most 3 open loans per borrower")
        if collateral_mP:
            self._debit_to_escrow(self.account(borrower), collateral_mP)
        loan = Loan(self._new_id("l_", borrower, principal_mP), borrower, principal_mP, interest_mP,
                    collateral_mP, term_blocks, terms_hash, self.height)
        if self.rules >= 2:
            loan.request_until = self.height + LOAN_REQUEST_BLOCKS
        self.loans[loan.loan_id] = loan
        return loan

    @transaction
    def cancel_loan_request(self, borrower: str, loan_id: str) -> None:
        loan = self.loans.get(loan_id)
        if loan is None or loan.borrower != borrower or loan.state != "REQUESTED":
            raise BoardError("no open loan request of yours with that id")
        loan.state = "CANCELLED"
        self._pay_from_escrow(borrower, loan.collateral_mP, lock=False)

    @transaction
    def fund_loan(self, lender: str, loan_id: str) -> Loan:
        loan = self.loans.get(loan_id)
        if loan is None or loan.state != "REQUESTED":
            raise BoardError("no open loan request with that id")
        if lender == loan.borrower:
            raise BoardError("cannot lend to yourself")
        self._transfer(lender, loan.borrower, loan.principal_mP)
        self._flow(lender, loan.borrower, loan.principal_mP, loan_id=loan_id)
        loan.state, loan.lender, loan.due = "ACTIVE", lender, self.height + loan.term_blocks
        return loan

    @transaction
    def repay_loan(self, borrower: str, loan_id: str) -> Loan:
        loan = self.loans.get(loan_id)
        if loan is None or loan.borrower != borrower or loan.state != "ACTIVE":
            raise BoardError("no active loan of yours with that id")
        self._settle_loan(loan)
        if loan.state != "REPAID":
            raise BoardError("not enough available balance to repay")
        return loan

    def _settle_loan(self, loan: Loan, at_due: bool = False) -> None:
        owed = loan.principal_mP + loan.interest_mP
        if self.account(loan.borrower).available_mP >= owed:
            self._transfer(loan.borrower, loan.lender, owed)
            self._flow(loan.borrower, loan.lender, owed, loan_id=loan.loan_id)
            self._pay_from_escrow(loan.borrower, loan.collateral_mP, lock=False)
            loan.state, loan.settled_height = "REPAID", self.height
            self._rep(loan.borrower, "loan_repaid", loan_id=loan.loan_id)
            if loan.interest_mP:
                self._rep(loan.lender, "loan_interest", loan_id=loan.loan_id, value_mP=loan.interest_mP)
        elif at_due and self.rules >= 2:
            # Default: the lender takes what the borrower can pay now, then up to the rest of the
            # debt plus a 10% penalty from the collateral; any collateral beyond that goes back.
            paid_now = self.account(loan.borrower).available_mP
            if paid_now > 0:
                self._transfer(loan.borrower, loan.lender, paid_now)
            claim = owed - paid_now + owed * LOAN_DEFAULT_PENALTY_BP // 10_000
            seized = min(loan.collateral_mP, claim)
            self._pay_from_escrow(loan.lender, seized, lock=False)
            self._pay_from_escrow(loan.borrower, loan.collateral_mP - seized, lock=False)
            self._flow(loan.borrower, loan.lender, paid_now + seized, loan_id=loan.loan_id)
            loan.state, loan.settled_height = "DEFAULTED", self.height
            self._rep(loan.borrower, "loan_default", loan_id=loan.loan_id, owed_mP=owed,
                      recovered_mP=paid_now + seized)
        elif at_due:
            self._pay_from_escrow(loan.lender, loan.collateral_mP, lock=False)
            loan.state, loan.settled_height = "DEFAULTED", self.height
            self._rep(loan.borrower, "loan_default", loan_id=loan.loan_id, owed_mP=owed,
                      recovered_mP=loan.collateral_mP)

    @transaction
    def transfer(self, sender: str, to: str, amount_mP: int, memo_hash: str) -> None:
        """Direct payment (01 doc 3.2). Nodes need their manager's approval first (Core-side)."""
        if to == sender:
            raise BoardError("cannot transfer to yourself")
        if to not in self.accounts or to in ("root", "lib"):
            raise BoardError(f"unknown or system account {to}")
        if amount_mP <= 0:
            raise BoardError("amount must be positive")
        self._transfer(sender, to, amount_mP)
        self._rep(to, "transfer_in", sender=sender, value_mP=amount_mP)

    # ---- savings ----

    @transaction
    def deposit_savings(self, owner: str, amount_mP: int, term_blocks: int) -> Deposit:
        if amount_mP < 1000:
            raise BoardError("deposit at least 1P")
        if not 720 <= term_blocks <= 17_280 * 30:
            raise BoardError("term must be 1 hour..30 days")
        if sum(1 for d in self.deposits.values() if d.owner == owner and d.state == "LOCKED") >= 10:
            raise BoardError("at most 10 open deposits")
        self._debit_to_escrow(self.account(owner), amount_mP)
        dep = Deposit(self._new_id("sv_", owner, amount_mP), owner, amount_mP, self.height,
                      self.height + term_blocks)
        self.deposits[dep.deposit_id] = dep
        return dep

    @transaction
    def withdraw_savings(self, owner: str, deposit_id: str) -> Deposit:
        dep = self.deposits.get(deposit_id)
        if dep is None or dep.owner != owner or dep.state != "LOCKED":
            raise BoardError("no locked deposit of yours with that id")
        self._pay_from_escrow(owner, dep.amount_mP, lock=False)
        self._burn_from_escrow(dep.earned_mP)  # early: the interest is forfeited
        dep.state = "WITHDRAWN_EARLY"
        return dep

    # ---- tool buy orders ----

    @transaction
    def create_tool_request(self, buyer: str, title: str, spec_hash: str, tests_hash: str,
                            reward_mP: int, deadline_blocks: int) -> ToolRequest:
        if reward_mP < self.params.min_reward_mP:
            raise BoardError(f"reward must be at least {self.params.min_reward_mP}mP")
        if not 120 <= deadline_blocks <= 17_280 * 7:
            raise BoardError("deadline must be between 120 blocks (10 min) and 7 days")
        self._debit_to_escrow(self.account(buyer), reward_mP)
        req = ToolRequest(self._new_id("tr_", buyer, title), buyer, title[:200], spec_hash, tests_hash,
                          reward_mP, self.height, self.height + deadline_blocks)
        self.tool_requests[req.request_id] = req
        return req

    @transaction
    def submit_tool(self, seller: str, request_id: str, tool_id: str, version: int) -> None:
        req = self.tool_requests.get(request_id)
        if req is None or req.state != "OPEN":
            raise BoardError("no open tool request with that id")
        if seller == req.buyer:
            raise BoardError("cannot fill your own request")
        listing = self._listing(tool_id, version)
        if listing.author != seller or listing.withdrawn:
            raise BoardError("submit a version of your own published tool")
        if any(x["tool_id"] == tool_id and x["version"] == version for x in req.submissions):
            raise BoardError("this version was already submitted")
        if sum(1 for x in req.submissions if x["seller"] == seller) >= 3:
            raise BoardError("at most 3 submissions per seller")
        req.submissions.append({"seller": seller, "tool_id": tool_id, "version": version,
                                "height": self.height, "passed": None})

    @transaction
    def verify_submission(self, verifier: str, request_id: str, tool_id: str, version: int,
                          passed: bool, report_hash: str) -> ToolRequest:
        """Root records the outcome of running the buyer's tests on a submission."""
        req = self.tool_requests.get(request_id)
        if req is None:
            raise BoardError("unknown tool request")
        sub = next((x for x in req.submissions if x["tool_id"] == tool_id and x["version"] == version
                    and x["passed"] is None), None)
        if sub is None:
            raise BoardError("no pending submission for that tool version")
        sub["passed"], sub["report_hash"] = bool(passed), report_hash
        if passed and req.state == "OPEN":
            req.state, req.winner, req.winner_tool = "FULFILLED", sub["seller"], [tool_id, version]
            self._pay_from_escrow(sub["seller"], req.reward_mP, lock=True)
            self.licenses.add((req.buyer, tool_id, version))
            self._rep(sub["seller"], "tool_licensed", tool_id=tool_id, version=version,
                      value_mP=req.reward_mP, via=request_id, **self._payer(req.buyer))
        return req

    def storage_leases(self, account: str) -> tuple[int, int]:
        """KB leased in and out by `account` that are still running."""
        lin = lout = 0
        for p in self.purchases.values():
            if p.state == "COMPLETED" and p.until is not None and self.height < p.until:
                kb = self.offers[p.offer_id].quantity
                if p.buyer == account:
                    lin += kb
                if p.seller == account:
                    lout += kb
        return lin, lout

    def has_license(self, account: str, tool_id: str, version: int) -> bool:
        listing = self.tools.get((tool_id, version))
        return (account, tool_id, version) in self.licenses and not (listing and listing.blocked)

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

    def _payer(self, payer: str) -> dict[str, str]:
        """Rules 2 record who paid each income, so issuance can net payments between a pair."""
        return {"payer": payer} if self.rules >= 2 else {}

    def _flow(self, payer: str, payee: str, amount: int, **extra: Any) -> None:
        """Rules 2: value moved between accounts outside the income kinds (loans), for netting."""
        if self.rules >= 2 and amount > 0:
            self._rep(payee, "flow", payer=payer, value_mP=amount, **extra)

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
        for b in self._live("bounties"):
            ev = self._transition(b, h)
            if ev:
                events.append(ev)
        if self.params.issuance and h > 0 and h % self.params.epoch_blocks == 0:
            ev = self._issue_epoch(h // self.params.epoch_blocks - 1)
            if ev:
                events.append(ev)
        for o in self._live("offers"):
            if o.state == "OPEN" and h >= o.expires:
                o.state = "EXPIRED"
        for pu in self._live("purchases"):
            if pu.state == "PENDING" and h >= pu.deliver_until:
                pu.state = "REFUNDED"
                self._pay_from_escrow(pu.buyer, pu.price_mP, lock=False)
                events.append(BoardEvent("purchase_refunded", None, {"purchase_id": pu.purchase_id}))
            elif pu.state == "DELIVERED" and pu.confirm_until is not None and h >= pu.confirm_until:
                # The buyer's Core never confirmed (crash between deliver and confirm): the seller
                # delivered on chain, so complete it; the buyer keeps the usual dispute window.
                self._complete_purchase(pu)
                events.append(BoardEvent("purchase_completed", None, {"purchase_id": pu.purchase_id,
                                                                      "reason": "confirm_timeout"}))
        for dep in self._live("deposits"):
            if dep.state == "LOCKED" and h >= dep.unlock:
                self._pay_from_escrow(dep.owner, dep.amount_mP + dep.earned_mP, lock=False)
                dep.state = "MATURED"
                events.append(BoardEvent("deposit_matured", None, {"deposit_id": dep.deposit_id,
                                                                   "owner": dep.owner, "earned_mP": dep.earned_mP}))
        for loan in self._live("loans"):
            if loan.state == "REQUESTED" and loan.request_until is not None and h >= loan.request_until:
                loan.state, loan.settled_height = "CANCELLED", h
                self._pay_from_escrow(loan.borrower, loan.collateral_mP, lock=False)
                events.append(BoardEvent("loan_expired", None, {"loan_id": loan.loan_id,
                                                                "borrower": loan.borrower}))
            elif loan.state == "ACTIVE" and h >= loan.due:
                self._settle_loan(loan, at_due=True)
                events.append(BoardEvent("loan_settled", None, {"loan_id": loan.loan_id, "state": loan.state,
                                                                "borrower": loan.borrower, "lender": loan.lender}))
        for sub in self._live("subscriptions"):
            if sub.state == "ACTIVE" and h >= sub.period_start + sub.period_blocks:
                self._close_period(sub)
                offer = self.offers[sub.offer_id]
                if offer.state == "OPEN" and self.account(sub.subscriber).available_mP >= sub.price_mP:
                    self._debit_to_escrow(self.account(sub.subscriber), sub.price_mP)
                    sub.period_start = h
                else:
                    sub.state = "ENDED"
                events.append(BoardEvent("subscription_period", None, {
                    "sub_id": sub.sub_id, "state": sub.state, "paid": sub.periods_paid,
                    "refunded": sub.periods_refunded}))
        for req in self._live("tool_requests"):
            if req.state == "OPEN" and h >= req.deadline:
                req.state = "EXPIRED"
                self._pay_from_escrow(req.buyer, req.reward_mP, lock=False)
                events.append(BoardEvent("tool_request_expired", None, {"request_id": req.request_id}))
        for prop in self._live("proposals"):
            if prop.state == "VOTING" and h >= prop.vote_until:
                events.append(self._tally(prop))
            elif prop.state == "TIMELOCK" and prop.timelock_until is not None and h >= prop.timelock_until:
                events.append(self._approve(prop))
        for d in self._live("disputes"):
            if d.state == "open" and h >= d.deadline:
                self._settle_dispute(d, "void", None)  # no ruling in time: bond returned
                events.append(BoardEvent("dispute_resolved", None, {
                    "dispute_id": d.dispute_id, "ruling": "void", "reason": "deadline"}))
        return events

    def _live(self, name: str) -> Any:
        """The objects of one collection the block loop may still act on, in creation order (the
        order the full collection would be walked in). New objects are picked up from the end of
        the collection; an object is dropped once it has left LIVE_IN_BLOCK_LOOP."""
        coll = getattr(self, name)
        index = self._live_index.setdefault(name, {})
        seen = self._live_seen.get(name, 0)
        if len(coll) > seen:
            for key in itertools.islice(coll, seen, None):
                index[key] = None
            self._live_seen[name] = len(coll)
        live = LIVE_IN_BLOCK_LOOP[name]
        for key in list(index):
            obj = coll[key]
            if live(obj):
                yield obj
            if not live(obj):
                del index[key]

    # ---- snapshots (gap review B3) ----

    def to_snapshot(self) -> dict[str, Any]:
        """The whole consensus state as plain JSON data, in collection order."""
        unknown = set(vars(self)) - set(SNAPSHOT_FIELDS) - RUNTIME_FIELDS
        if unknown:
            raise BoardError(f"snapshot does not cover board fields {sorted(unknown)}")
        rows = lambda coll: [asdict(v) for v in coll.values()]  # noqa: E731
        return {
            "params": asdict(self.params), "rules": self.rules, "height": self.height,
            "escrow_mP": self.escrow_mP, "burned_mP": self.burned_mP, "issued_mP": self.issued_mP,
            "_seq": self._seq, "rep_events": self.rep_events,
            "_heartbeats": sorted([a, e] for a, e in self._heartbeats),
            "licenses": sorted(list(x) for x in self.licenses),
            **{name: rows(getattr(self, name)) for name in (
                "accounts", "bounties", "answers", "tools", "disputes", "proposals", "tool_requests",
                "offers", "purchases", "subscriptions", "loans", "deposits", "ratings")},
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Replace this board's state with a snapshot (in place: the network keeps its reference)."""
        params = dict(data["params"])
        params["commit_blocks_range"] = tuple(params["commit_blocks_range"])
        self.params = Params(**params)
        self.rules, self.height = data["rules"], data["height"]
        self.escrow_mP, self.burned_mP, self.issued_mP = data["escrow_mP"], data["burned_mP"], data["issued_mP"]
        self._seq, self.rep_events = data["_seq"], data["rep_events"]
        self._heartbeats = {(a, e) for a, e in data["_heartbeats"]}
        self.licenses = {tuple(x) for x in data["licenses"]}
        self.accounts = {a["id"]: Account(**{**a, "locks": [tuple(lk) for lk in a["locks"]]})
                         for a in data["accounts"]}

        def bounty(b: dict[str, Any]) -> Bounty:
            return Bounty(**{**b, "state": BountyState(b["state"]),
                             "outcome": BountyState(b["outcome"]) if b["outcome"] else None})

        self.bounties = {b["bounty_id"]: bounty(b) for b in data["bounties"]}
        self.answers = {a["answer_id"]: Answer(**a) for a in data["answers"]}
        self.tools = {(t["tool_id"], t["version"]): ToolListing(**t) for t in data["tools"]}
        self.disputes = {d["dispute_id"]: Dispute(**d) for d in data["disputes"]}
        self.proposals = {p["proposal_id"]: Proposal(**p) for p in data["proposals"]}
        self.tool_requests = {r["request_id"]: ToolRequest(**r) for r in data["tool_requests"]}
        self.offers = {o["offer_id"]: Offer(**o) for o in data["offers"]}
        self.purchases = {p["purchase_id"]: Purchase(**p) for p in data["purchases"]}
        self.subscriptions = {s["sub_id"]: Subscription(**s) for s in data["subscriptions"]}
        self.loans = {x["loan_id"]: Loan(**x) for x in data["loans"]}
        self.deposits = {d["deposit_id"]: Deposit(**d) for d in data["deposits"]}
        self.ratings = {r["rating_id"]: Rating(**r) for r in data["ratings"]}
        self._live_index, self._live_seen = {}, {}

    def _issue_epoch(self, epoch: int) -> BoardEvent | None:
        """End-of-epoch issuance (03 doc 2.2): 60% uptime pool shared equally by accounts with a
        heartbeat in the epoch, 40% contribution pool by confirmed income (max 50% each).
        A pool with nobody eligible is not issued."""
        p = self.params
        start, end = epoch * p.epoch_blocks, (epoch + 1) * p.epoch_blocks
        reward = p.epoch_blocks * (p.block_reward_mP >> (start // p.reward_halving_blocks))
        if reward <= 0:
            return None
        savers = {d.deposit_id: d for d in self.deposits.values() if d.state == "LOCKED"}
        savings_pool = reward // 10 if savers else 0  # with no deposits the pool is not issued
        uptime_pool = reward * 55 // 100 if savers else reward * 6 // 10
        contrib_pool = reward - uptime_pool - (reward // 10 if savers else 0)
        if self.rules >= 2:
            first = self._events_from(min(start, epoch * 720))
            alive = sorted({e["account"] for e in self.rep_events[first:]
                            if e["kind"] == "heartbeat" and e.get("epoch") == epoch})
            income = self._netted_income(epoch, start, end)
        else:
            alive = sorted({e["account"] for e in self.rep_events
                            if e["kind"] == "heartbeat" and e.get("epoch") == epoch})
            income = {}
            for e in self.rep_events:
                if start <= e["height"] < end and e["kind"] in INCOME_KINDS:
                    income[e["account"]] = income.get(e["account"], 0) + e.get("value_mP", 0)
        paid: dict[str, int] = {}
        if alive:
            share = uptime_pool // len(alive)
            for a in alive:
                paid[a] = paid.get(a, 0) + share
        total_income = sum(income.values())
        if total_income > 0:
            for a, inc in sorted(income.items()):
                share = min(contrib_pool * inc // total_income, contrib_pool // 2)
                paid[a] = paid.get(a, 0) + share
        total_saved = sum(d.amount_mP for d in savers.values())
        saved_paid = 0
        for dep_id, d in sorted(savers.items()):
            share = savings_pool * d.amount_mP // total_saved if total_saved else 0
            d.earned_mP += share  # held in escrow until maturity
            self.escrow_mP += share
            self.issued_mP += share
            saved_paid += share
        for a, amount in paid.items():
            if amount > 0 and a in self.accounts:
                self.accounts[a].balance_mP += amount
                self.issued_mP += amount
        if saved_paid:
            paid = {**paid, "savings": saved_paid}
        if not paid:
            return None
        return BoardEvent("epoch_issued", None, {"epoch": epoch, "paid": paid})

    def _events_from(self, height: int) -> int:
        """Index of the first reputation event at or after `height` (events are height-ordered)."""
        return bisect.bisect_left(self.rep_events, height, key=lambda e: e["height"])

    def _netted_income(self, epoch: int, start: int, end: int) -> dict[str, int]:
        """Contribution income for one epoch with payments between the same two accounts netted
        over a trailing day (rules 2). Each payer->payee income counts only as far as it exceeds
        what flowed back the other way (any income, transfer or loan flow) and what was already
        credited for that pair in the window, so passing points back and forth earns nothing.
        Income events without a payer (made under rules 1) count as before."""
        wstart = max(0, end - INCOME_WINDOW_EPOCHS * self.params.epoch_blocks)
        inc_w: dict[tuple[str, str], int] = {}
        moved_w: dict[tuple[str, str], int] = {}
        credited_w: dict[tuple[str, str], int] = {}
        inc_e: dict[tuple[str, str], int] = {}
        income: dict[str, int] = {}
        for e in self.rep_events[self._events_from(wstart):]:
            if e["height"] >= end:
                break
            kind, v = e["kind"], e.get("value_mP", 0)
            if kind in INCOME_KINDS:
                if "payer" not in e:
                    if e["height"] >= start:
                        income[e["account"]] = income.get(e["account"], 0) + v
                    continue
                pair = (e["payer"], e["account"])
                inc_w[pair] = inc_w.get(pair, 0) + v
                moved_w[pair] = moved_w.get(pair, 0) + v
                if e["height"] >= start:
                    inc_e[pair] = inc_e.get(pair, 0) + v
            elif kind in ("transfer_in", "flow"):
                pair = (e.get("payer") or e.get("sender"), e["account"])
                moved_w[pair] = moved_w.get(pair, 0) + v
            elif kind == "income_credit":
                pair = (e["payer"], e["account"])
                credited_w[pair] = credited_w.get(pair, 0) + v
        for (payer, payee), v in sorted(inc_e.items()):
            net = inc_w[(payer, payee)] - moved_w.get((payee, payer), 0) - credited_w.get((payer, payee), 0)
            counted = max(0, min(v, net))
            if counted:
                income[payee] = income.get(payee, 0) + counted
                self._rep(payee, "income_credit", payer=payer, value_mP=counted, epoch=epoch)
        return income

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
            for a in unrevealed:
                self._burn_from_escrow(self._bond_of(self.answers[a]))
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
                    self._pay_from_escrow(w, share, lock=True, bounty=b)
                    self._rep(w, "answer_auto", bounty_id=b.bounty_id, value_mP=share, **self._payer(b.asker))
                self._burn_from_escrow(b.reward_mP - share * len(winners))
                self._rep(b.asker, "judge_timeout", bounty_id=b.bounty_id)
                return self._settle(b, BountyState.AUTO_SPLIT)
            self._pay_from_escrow(b.asker, pool, lock=True, bounty=b)
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
