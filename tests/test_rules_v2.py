"""Consensus rules 2: the money fixes from the 2026-10-12 gap review, and the rules upgrade that
keeps old blocks replaying under the rules they were made with."""

import random

import pytest

from aichain import audit
from aichain.board import (CONFIRM_BLOCKS, LOAN_REQUEST_BLOCKS, Board, BoardError, BountyState,
                           Params, commitment_of)
from aichain.chain import store as chain_store
from aichain.chain.store import ChainStore, attach, replay
from aichain.common import sha256_hex

P = 1000
SECRET = "plaintext"


def params(**kw):
    base = dict(commit_blocks=12, reveal_blocks=6, judge_blocks=10, dispute_blocks=20,
                dispute_resolve_blocks=50, issuance=False)
    base.update(kw)
    return Params(**base)


def books_ok(b):
    assert b.total_supply_check()
    assert b.escrow_obligations() == b.escrow_mP
    assert audit.run(b) == []


def accounts(b, *names, amount=1000 * P):
    for n in names:
        b.open_account(n, amount)


def bought(b, ok=None):
    o = b.create_offer("s", "info", "t", "terms", 3 * P, sha256_hex(SECRET), 720)
    pu = b.buy_offer("buyer", o.offer_id)
    b.deliver_offer("s", pu.purchase_id, "c")
    if ok is not None:
        b.confirm_delivery("buyer", pu.purchase_id, ok, "r")
    return pu


# ---- A1: a deal dispute nobody rules on ----

@pytest.mark.parametrize("rules", [1, 2])
def test_void_deal_dispute(rules):
    b = Board(params(), rules=rules)
    accounts(b, "s", "buyer")
    pu = bought(b, ok=False)
    assert pu.state == "DISPUTED"
    b.advance(50)  # ruling deadline passes: void
    if rules == 1:  # the old bug, kept for replaying old blocks: the price stays in escrow
        assert pu.state == "DISPUTED" and b.escrow_mP == 3 * P
    else:
        assert pu.state == "REFUNDED" and b.escrow_mP == 0
        assert b.accounts["buyer"].balance_mP == 1000 * P
    books_ok(b)


def test_rejected_deal_dispute_pays_seller():
    b = Board(params())
    accounts(b, "s", "buyer")
    pu = bought(b, ok=False)
    d = next(iter(b.disputes.values()))
    b.resolve_dispute("root", d.dispute_id, "reject", "x")
    assert pu.state == "COMPLETED" and b.accounts["s"].balance_mP == 1003 * P
    books_ok(b)


# ---- A2: bond parameter changed between commit and reveal ----

def test_bond_change_cannot_mint_or_burn():
    b = Board(params(commit_bond_mP=500))
    accounts(b, "asker", "a1")
    bounty = b.create_bounty("asker", "q", 20 * P, [])
    a = b.commit_answer("a1", bounty.bounty_id, commitment_of("h", "s", "a1"))
    b.params.commit_bond_mP = 10_000  # as a param proposal would
    b.advance(12)
    b.reveal_answer("a1", a.answer_id, "h", "s")
    assert b.accounts["a1"].balance_mP == 1000 * P  # exactly the 0.5P bond came back
    assert b.escrow_mP == 20 * P
    books_ok(b)


# ---- A3: disputes hold the disputed amount ----

def paid_bounty(b):
    accounts(b, "asker", "a1", "a2", "w")
    bounty = b.create_bounty("asker", "q", 20 * P, [])
    ans = {n: b.commit_answer(n, bounty.bounty_id, commitment_of("h", "s", n)) for n in ("a1", "a2")}
    b.advance(12)
    for n, a in ans.items():
        b.reveal_answer(n, a.answer_id, "h", "s")
    b.advance(6)
    b.accept_answer("asker", bounty.bounty_id, ans["a1"].answer_id)
    return bounty, ans


def test_upheld_answer_dispute_recovers_after_lock_expired():
    b = Board(params())
    bounty, ans = paid_bounty(b)
    d = b.open_dispute("w", "answer", ans["a1"].answer_id, "claim")
    assert d.held_mP == 20 * P
    b.advance(30)  # the 20-block settlement lock ends while the dispute is still open
    # The author tries to move the money away; it is not there to move.
    with pytest.raises(BoardError):
        b.transfer("a1", "a2", 1005 * P, "m")
    b.resolve_dispute("root", d.dispute_id, "uphold", "r")
    assert b.accounts["w"].balance_mP == 1004 * P  # bond back + 20% finder reward (4P)
    assert b.accounts["asker"].balance_mP == 996 * P  # paid 20P, recovered 16P
    books_ok(b)


@pytest.mark.parametrize("ruling", ["reject", "void"])
def test_dispute_hold_is_returned(ruling):
    b = Board(params())
    bounty, ans = paid_bounty(b)
    d = b.open_dispute("w", "answer", ans["a1"].answer_id, "claim")
    before = b.accounts["a1"].balance_mP
    b.resolve_dispute("root", d.dispute_id, ruling, "r")
    assert b.accounts["a1"].balance_mP >= before + 20 * P
    books_ok(b)


# ---- A4: pairwise netting of contribution income ----

def epoch_board():
    b = Board(Params(issuance=True, epoch_blocks=720))
    accounts(b, "a", "b", "h1", "h2")
    return b


def heartbeat_all(b):
    for n in ("a", "b", "h1", "h2"):
        b.heartbeat(n, b.height // 720)


def info_sale(b, seller, buyer, price):
    o = b.create_offer(seller, "info", "t", "terms", price, sha256_hex(SECRET + seller + buyer + str(b.height)), 720)
    pu = b.buy_offer(buyer, o.offer_id)
    b.deliver_offer(seller, pu.purchase_id, "c")
    b.confirm_delivery(buyer, pu.purchase_id, True, "r")


def test_wash_trading_earns_no_contribution():
    b = epoch_board()
    heartbeat_all(b)
    for _ in range(3):  # a and b pass 100P back and forth
        info_sale(b, "a", "b", 100 * P)
        info_sale(b, "b", "a", 100 * P)
    info_sale(b, "h1", "h2", 10 * P)  # one honest sale
    start = {n: b.accounts[n].balance_mP for n in ("a", "b", "h1")}
    b.advance(720 - b.height)
    gain = {n: b.accounts[n].balance_mP - start[n] for n in start}
    # Everyone gets the same uptime share; only the honest seller gets contribution income.
    assert gain["a"] == gain["b"] < gain["h1"]
    credits = [e for e in b.rep_events if e["kind"] == "income_credit"]
    assert {e["account"] for e in credits} == {"h1"}
    books_ok(b)


def test_alternating_epochs_credit_once_per_window():
    b = epoch_board()
    credited = []
    for epoch in range(4):
        heartbeat_all(b)
        seller, buyer = ("a", "b") if epoch % 2 == 0 else ("b", "a")
        info_sale(b, seller, buyer, 50 * P)
        b.advance(720 * (epoch + 1) - b.height)
        credited.append(sum(e["value_mP"] for e in b.rep_events
                            if e["kind"] == "income_credit" and e["epoch"] == epoch))
    assert credited[0] == 50 * P and sum(credited[1:]) == 0


def test_transfer_back_cancels_income():
    b = epoch_board()
    heartbeat_all(b)
    info_sale(b, "a", "b", 100 * P)
    b.transfer("a", "b", 100 * P, "m")
    b.advance(720)
    assert not [e for e in b.rep_events if e["kind"] == "income_credit"]


# ---- A5: loans ----

def test_overcollateralised_default_returns_the_rest():
    b = Board(params())
    accounts(b, "lender")
    accounts(b, "borrower", amount=200 * P)
    loan = b.create_loan_request("borrower", 10 * P, 1 * P, 200 * P, 720, "t")
    b.fund_loan("lender", loan.loan_id)
    b.accounts["borrower"].balance_mP -= 10 * P  # spent the loan
    b.burned_mP += 10 * P
    b.advance(720)
    assert loan.state == "DEFAULTED"
    # Owed 11P + 10% penalty (1.1P) taken from the collateral; the other 187.9P comes back.
    assert b.accounts["lender"].balance_mP == 1000 * P - 10 * P + 12_100
    assert b.accounts["borrower"].balance_mP == 200 * P - 12_100
    books_ok(b)


def test_unfunded_loan_request_expires():
    b = Board(params())
    accounts(b, "borrower")
    loan = b.create_loan_request("borrower", 10 * P, 0, 5 * P, 720, "t")
    b.advance(LOAN_REQUEST_BLOCKS)
    assert loan.state == "CANCELLED" and b.accounts["borrower"].balance_mP == 1000 * P
    books_ok(b)


# ---- A6: delivered but never confirmed ----

def test_unconfirmed_delivery_completes():
    b = Board(params())
    accounts(b, "s", "buyer")
    pu = bought(b)
    assert pu.state == "DELIVERED"
    b.advance(CONFIRM_BLOCKS)
    assert pu.state == "COMPLETED" and b.accounts["s"].balance_mP == 1003 * P
    books_ok(b)


# ---- C3, C4, C6 ----

def test_close_implementing_proposal():
    b = Board(Params(issuance=False, vote_blocks=720, timelock_blocks=360))
    for n in ("n1", "n2", "n3", "n4"):
        b.open_account(n, 1000 * P)
    prop = b.create_proposal("n1", "upgrade", "t", "spec", [])
    for n in ("n1", "n2", "n3"):
        b.vote(n, prop.proposal_id, "yes", "r")
    b.advance(720 + 360)
    assert prop.state == "IMPLEMENTING"
    with pytest.raises(BoardError):
        b.close_proposal("root", prop.proposal_id, "done", "n")
    b.close_proposal("root", prop.proposal_id, "implemented", "note")
    assert prop.state == "IMPLEMENTED"


def test_blocked_tool_stops_sales_and_calls():
    b = Board(params())
    accounts(b, "s", "u", "v")
    b.publish_tool("s", "s.t", 1, "code", 1 * P, 100)
    b.license_tool("u", "s.t", 1)
    b.block_tool("root", "s.t", 1, "malware")
    assert not b.has_license("u", "s.t", 1)
    with pytest.raises(BoardError):
        b.pay_tool_use("u", "s.t", 1, 1)
    with pytest.raises(BoardError):
        b.license_tool("v", "s.t", 1)


def test_bounty_options():
    b = Board(params())
    accounts(b, "asker", "a1")
    bounty = b.create_bounty("asker", "q", 2 * P, [], reveal_blocks=30, judge_blocks=40,
                             min_rep_e6=900_000)
    assert bounty.reveal_until == bounty.commit_until + 30
    assert bounty.judge_until == bounty.reveal_until + 40
    with pytest.raises(BoardError, match="reputation"):  # a new node starts at 0.45
        b.commit_answer("a1", bounty.bounty_id, "c")
    old = Board(params(), rules=1)
    accounts(old, "asker")
    with pytest.raises(BoardError):
        old.create_bounty("asker", "q", 2 * P, [], min_rep_e6=1)


# ---- escrow invariant under random activity ----

def test_escrow_matches_obligations_under_random_activity():
    rnd = random.Random(7)
    b = Board(params(issuance=True, epoch_blocks=60))
    names = ["n1", "n2", "n3", "n4"]
    accounts(b, *names)
    for step in range(600):
        who, other = rnd.sample(names, 2)
        try:
            op = rnd.randrange(9)
            if op == 0:
                b.create_bounty(who, "q", rnd.randint(1, 5) * P, [])
            elif op == 1:
                open_b = [x for x in b.bounties.values() if x.state == BountyState.OPEN and x.asker != who]
                if open_b:
                    b.commit_answer(who, rnd.choice(open_b).bounty_id, commitment_of("h", "s", who))
            elif op == 2:
                for a in list(b.answers.values()):
                    if not a.revealed and b.bounties[a.bounty_id].state == BountyState.REVEAL:
                        b.reveal_answer(a.author, a.answer_id, "h", "s")
            elif op == 3:
                info_sale(b, who, other, rnd.randint(1, 3) * P)
            elif op == 4:
                pu = bought_by(b, who, other)
                b.confirm_delivery(other, pu.purchase_id, False, "r")
            elif op == 5:
                loan = b.create_loan_request(who, 5 * P, 500, rnd.randint(0, 8) * P, 720, "t")
                if rnd.random() < 0.7:
                    b.fund_loan(other, loan.loan_id)
            elif op == 6:
                b.deposit_savings(who, rnd.randint(1, 20) * P, 720)
            elif op == 7:
                paid = [a for a in b.answers.values() if b.bounties[a.bounty_id].state == BountyState.AUTO_SPLIT]
                if paid:
                    a = rnd.choice(paid)
                    b.open_dispute(rnd.choice([n for n in names if n != a.author]), "answer", a.answer_id, "c")
            else:
                for d in [d for d in b.disputes.values() if d.state == "open"][:1]:
                    b.resolve_dispute("root", d.dispute_id, rnd.choice(["uphold", "reject", "void"]), "r")
        except BoardError:
            pass
        b.advance(rnd.randint(1, 30))
        assert b.escrow_obligations() == b.escrow_mP, f"step {step}"
        assert b.total_supply_check()
        assert all(a.balance_mP >= 0 for a in b.accounts.values())


def bought_by(b, seller, buyer):
    o = b.create_offer(seller, "info", "t", "terms", 2 * P, sha256_hex(f"{seller}{buyer}{b.height}"), 720)
    pu = b.buy_offer(buyer, o.offer_id)
    b.deliver_offer(seller, pu.purchase_id, "c")
    return pu


# ---- A8: the rules upgrade on a real chain log ----

def test_old_log_replays_under_rules_1_then_upgrades(tmp_path, monkeypatch):
    db = tmp_path / "chain.db"
    # A chain written by code that only knew rules 1: no upgrade transaction in its log.
    monkeypatch.setattr(chain_store, "RULES_LATEST", 1)
    old = Board(params())
    attach(old, ChainStore(db))
    assert old.rules == 1
    accounts(old, "s", "buyer")
    stuck = bought(old, ok=False)
    old.advance(50)  # rules 1: the void dispute leaves the price in escrow
    assert stuck.state == "DISPUTED"
    root_before = old.state_root()

    # New code starts on the same log: replays it under rules 1, then upgrades at the tip.
    monkeypatch.undo()
    new = Board(params())
    stats = attach(new, ChainStore(db))
    assert stats["rules_upgraded_to"] == 2 and new.rules == 2
    assert new.purchases[stuck.purchase_id].state == "DISPUTED"
    new.rules = 1
    assert new.state_root() == root_before  # same state, same root as the old code computed
    new.rules = 2

    pu = bought(new, ok=False)
    new.advance(50)
    assert pu.state == "REFUNDED"  # rules 2 from the upgrade on

    again = Board(params())
    attach(again, ChainStore(db))
    assert again.rules == 2 and again.state_root() == new.state_root()
    assert again.purchases[stuck.purchase_id].state == "DISPUTED"
    assert again.purchases[pu.purchase_id].state == "REFUNDED"


def test_fresh_chain_starts_upgraded(tmp_path):
    db = tmp_path / "chain.db"
    b = Board(params())
    attach(b, ChainStore(db))
    assert b.rules == 2
    accounts(b, "x")
    b.advance(3)
    rebuilt = Board(params())
    replay(rebuilt, ChainStore(db))
    assert rebuilt.rules == 2 and rebuilt.state_root() == b.state_root()
