import json

from aichain.gateways.web import FakeWebProvider, SearchGateway
from aichain.mastermind.tools import ALWAYS, EVENT_TOOLS, tool_specs
from aichain.node import QueuedEvent
from aichain.sim.factory import build_network
from aichain.workshop.registry import Workshop


def test_tools_are_scoped_by_event():
    net = build_network("scripted", web=SearchGateway(FakeWebProvider()), workshop=Workshop())
    n2 = net.nodes["n2"]
    all_tools = n2.tool_names()
    n2.current_event = QueuedEvent(3, 0, "bounty_new", {})
    scoped = n2.tool_names()
    n2.current_event = None
    assert "answer" in scoped and "loans" not in scoped and "propose" not in scoped
    assert set(scoped) <= set(all_tools) and ALWAYS & set(all_tools) <= set(scoped)
    size = lambda names: len(json.dumps([t.parameters for t in tool_specs(names)]))
    assert size(scoped) < size(all_tools) / 2
    # A tool outside the scope is refused during that event.
    n2.current_event = QueuedEvent(3, 0, "bounty_new", {})
    assert "지금 쓸 수 없는 도구" in n2._exec_tool("loans", "{}")
    n2.current_event = None


def test_economy_ticks_are_staggered_and_summarise_the_node():
    net = build_network("scripted")
    net.economy_interval_blocks = 80
    net.advance(80)
    ticks = [(e["height"], e["node"]) for e in net.events
             if e["event"] == "episode_start" and e["kind"] == "economy_tick"]
    assert sorted(n for _, n in ticks) == ["n1", "n2", "n3", "n4"] and len({h for h, _ in ticks}) == 4
    summary = net.nodes["n1"]._economy_summary()
    assert "오늘 남은 할당량" in summary and "대출 요청" in summary and "저장 공간" in summary
    assert "economy_tick" in EVENT_TOOLS and "savings" in EVENT_TOOLS["economy_tick"]


def test_ratings_for_every_kind_of_deal():
    import pytest
    from aichain.board import Board, BoardError, Params
    from aichain.common import sha256_hex

    b = Board(Params(issuance=False))
    for n in ("a", "b", "c"):
        b.open_account(n, 100_000)
    o = b.create_offer("a", "info", "t", "h", 1000, sha256_hex("x"), 720)
    pu = b.buy_offer("b", o.offer_id)
    with pytest.raises(BoardError):
        b.rate("b", "a", 1, pu.purchase_id)  # not settled yet
    b.deliver_offer("a", pu.purchase_id, "c")
    b.confirm_delivery("b", pu.purchase_id, True, "r")
    b.rate("b", "a", 2, pu.purchase_id)
    loan = b.create_loan_request("c", 5000, 0, 0, 720, "t")
    b.fund_loan("a", loan.loan_id)
    b.repay_loan("c", loan.loan_id)
    b.rate("a", "c", 1, loan.loan_id)
    with pytest.raises(BoardError):
        b.rate("b", "c", 1, loan.loan_id)  # b was not part of the loan
    assert len(b.ratings) == 2


def test_tool_disputes_refund_and_withdraw():
    from aichain.board import Board, Params
    from aichain.reputation import for_board

    b = Board(Params(issuance=False))
    for n in ("author", "buyer"):
        b.open_account(n, 100_000)
    b.publish_tool("author", "author.x", 1, "h", 20_000, 0)
    b.license_tool("buyer", "author.x", 1)
    d = b.open_dispute("buyer", "tool", "author.x@1", "output differs from the description")
    assert d.bond_mP == 2000
    b.resolve_dispute("root", d.dispute_id, "uphold", "")
    assert b.accounts["buyer"].balance_mP == 100_000 and b.tools[("author.x", 1)].withdrawn
    assert for_board(b, "author").terms_e6["T"] < 500_000 and b.total_supply_check()
