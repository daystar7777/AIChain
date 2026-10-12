import json

import pytest

from aichain.board import Board, BoardError, Params
from aichain.chain.store import ChainStore, attach, replay
from aichain.reputation import for_board
from aichain.sim.factory import build_network


def board():
    b = Board(Params(issuance=False))
    b.open_account("lender", 100_000)
    b.open_account("borrower", 10_000)
    return b


def test_auto_repay_at_due():
    b = board()
    loan = b.create_loan_request("borrower", 20_000, 2_000, 5_000, 720, "t")
    assert b.accounts["borrower"].available_mP == 5_000
    b.fund_loan("lender", loan.loan_id)
    assert b.accounts["borrower"].balance_mP == 25_000 and b.accounts["borrower"].available_mP == 25_000
    b.accounts["borrower"].balance_mP += 2_000  # borrower earned the interest meanwhile
    b.issued_mP += 2_000
    b.advance(720)
    assert loan.state == "REPAID"
    assert b.accounts["lender"].balance_mP == 102_000 and b.accounts["borrower"].balance_mP == 10_000
    assert any(e["kind"] == "loan_interest" for e in b.rep_events)
    assert b.total_supply_check()


def test_default_gives_collateral_and_hurts_reliability():
    b = board()
    loan = b.create_loan_request("borrower", 20_000, 2_000, 5_000, 720, "t")
    b.fund_loan("lender", loan.loan_id)
    b.accounts["borrower"].balance_mP -= 20_000  # spent the loan
    b.burned_mP += 20_000
    before = for_board(b, "borrower").terms_e6["J"]
    b.advance(720)
    assert loan.state == "DEFAULTED"
    assert b.accounts["lender"].balance_mP == 85_000  # lent 20P, got 5P collateral
    assert for_board(b, "borrower").terms_e6["J"] < before
    assert b.total_supply_check()


def test_rules_and_early_repay():
    b = board()
    with pytest.raises(BoardError):
        b.create_loan_request("borrower", 20_000, 10_000, 0, 720, "t")  # >30% interest
    with pytest.raises(BoardError):
        b.create_loan_request("borrower", 500, 0, 0, 720, "t")  # < 1P
    loan = b.create_loan_request("borrower", 5_000, 500, 0, 720, "t")
    with pytest.raises(BoardError):
        b.fund_loan("borrower", loan.loan_id)
    b.fund_loan("lender", loan.loan_id)
    b.repay_loan("borrower", loan.loan_id)
    assert loan.state == "REPAID" and b.accounts["borrower"].balance_mP == 9_500
    other = b.create_loan_request("borrower", 5_000, 0, 1_000, 720, "t")
    b.cancel_loan_request("borrower", other.loan_id)
    assert other.state == "CANCELLED" and b.accounts["borrower"].available_mP == 9_500


def test_loans_are_signed_and_replay(tmp_path):
    db = tmp_path / "c.db"
    live = Board(Params(issuance=False))
    attach(live, ChainStore(db))
    live.open_account("lender", 100_000)
    live.open_account("borrower", 10_000)
    loan = live.create_loan_request("borrower", 5_000, 500, 1_000, 720, "t")
    live.fund_loan("lender", loan.loan_id)
    live.advance(720)
    rebuilt = Board(Params(issuance=False))
    replay(rebuilt, ChainStore(db))
    assert rebuilt.state_root() == live.state_root() and rebuilt.loans[loan.loan_id].state == "REPAID"


def test_nodes_borrow_and_lend_through_tools():
    net = build_network("scripted")
    n1, n2 = net.nodes["n1"], net.nodes["n2"]
    out = n1._exec_tool("loan_request", json.dumps({"principal_P": 30, "interest_pct": 10, "term_hours": 1,
                                                     "collateral_P": 5, "purpose": "도구 구매 자금"}))
    assert "대출 요청" in out
    loan = next(iter(net.board.loans.values()))
    assert all(n.has_pending for n in (net.nodes["n2"], net.nodes["n3"], net.nodes["n4"]))  # announced
    assert "도구 구매 자금" in n2._exec_tool("loans", json.dumps({"action": "list"}))
    assert "빌려줬다" in n2._exec_tool("loans", json.dumps({"action": "fund", "loan_id": loan.loan_id}))
    net.advance(720)
    assert loan.state == "REPAID"
    assert any(m.source == f"chain:{loan.loan_id}" for m in n2.memory.records)
