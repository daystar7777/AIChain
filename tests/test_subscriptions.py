import json

import pytest

from aichain.board import Board, BoardError, Params
from aichain.gateways.llm.scripted import tool_response
from aichain.sim.brains import persona_brain
from aichain.sim.factory import build_network

P = 720  # one-hour period


def board():
    b = Board(Params(issuance=False))
    b.open_account("s", 100_000)
    b.open_account("u", 10_000)
    return b


def test_periods_pay_only_when_delivered_and_renew():
    b = board()
    o = b.create_offer("s", "subscription", "AI 동향", "t", 3000, None, 17_280, period_blocks=P)
    with pytest.raises(BoardError):
        b.buy_offer("u", o.offer_id)
    sub = b.subscribe("u", o.offer_id)
    with pytest.raises(BoardError):
        b.subscribe("u", o.offer_id)
    b.sub_deliver("s", sub.sub_id, "c1")
    b.advance(P)  # period 1 delivered -> paid; renews
    assert sub.periods_paid == 1 and b.accounts["s"].balance_mP == 103_000
    assert b.accounts["u"].balance_mP == 4_000 and sub.state == "ACTIVE"  # 3P paid + 3P escrowed
    b.advance(P)  # period 2 nothing delivered -> refunded; renews again
    assert sub.periods_refunded == 1 and b.accounts["u"].balance_mP == 4_000
    sub2 = b.cancel_subscription("u", sub.sub_id)
    assert sub2.state == "ENDED" and b.accounts["u"].balance_mP == 7_000
    assert b.total_supply_check()


def test_subscription_ends_when_the_subscriber_cannot_pay():
    b = board()
    o = b.create_offer("s", "subscription", "t", "t", 6000, None, 17_280, period_blocks=P)
    sub = b.subscribe("u", o.offer_id)
    b.sub_deliver("s", sub.sub_id, "c")
    b.advance(P)
    assert sub.state == "ENDED" and b.accounts["u"].balance_mP == 4_000


def test_note_updates_are_pushed_encrypted_to_subscribers():
    holder = {}

    def brain(request):
        tools = [m for m in request.messages if m.role == "tool"]
        user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if "idle_tick" not in user:
            return persona_brain(request)
        if tools:
            return tool_response(("done", {"summary": "ok"}))
        net = holder["net"]
        if request.node_id == "n4" and not net.board.offers:
            return tool_response(("offer_create", {"kind": "subscription", "title": "과학 소식 구독",
                                                   "summary": "과학 소식 노트 갱신본", "note_topic": "과학 소식",
                                                   "price_P": 3, "period_hours": 1}))
        if request.node_id == "n4":
            return tool_response(("note", {"topic": "과학 소식", "summary": "v2: 새 발견 추가", "confidence": 0.7}))
        return tool_response(("offer_buy", {"offer_id": next(iter(net.board.offers))}))

    net = build_network("scripted", brain=brain)
    holder["net"] = net
    net.nodes["n4"].memory.remember(0, "note", "self:research", 0.3, "[과학 소식] v1", ["과학 소식"],
                                    topic="과학 소식", data={"summary": "v1: 핵융합", "facts": [], "confidence": 0.6})
    for nid in ("n4", "n1", "n4"):
        net.nodes[nid].enqueue("idle_tick", {})
        net.run_until_idle()
    (sub,) = net.board.subscriptions.values()
    assert sub.deliveries == 2  # the current note at subscription, then the v2 update
    got = [m.content for m in net.nodes["n1"].memory.records if m.source == "deal:n4"]
    assert "v1: 핵융합" in got[0] and "v2: 새 발견 추가" in got[1]
    assert not any("새 발견" in json.dumps(o, ensure_ascii=False) for o in net.content._items.values())
    net.advance(P)
    assert sub.periods_paid == 1


# ---- A7: deliveries only on change, renewal notices, renewals within the Policy Guard limits ----

def subscribed_net():
    net = build_network("scripted")
    seller, buyer = net.nodes["n4"], net.nodes["n1"]
    note = lambda summary: seller._exec_tool("note", json.dumps({"topic": "과학 소식", "summary": summary,
                                                                  "confidence": 0.7}))
    assert "지식 노트" in note("v1: 핵융합")
    out = seller._exec_tool("offer_create", json.dumps({
        "kind": "subscription", "title": "과학 소식 구독", "summary": "과학 소식 노트 갱신본",
        "note_topic": "과학 소식", "price_P": 3, "period_hours": 1}))
    assert "구독 상품" in out
    out = buyer._exec_tool("offer_buy", json.dumps({"offer_id": next(iter(net.board.offers))}))
    assert "구독 시작" in out
    (sub,) = net.board.subscriptions.values()
    return net, sub, note


def test_resaving_an_unchanged_note_is_not_a_delivery():
    net, sub, note = subscribed_net()
    assert sub.deliveries == 1  # the current note at subscription
    net.advance(P)  # period 1: delivered -> paid
    assert sub.periods_paid == 1
    note("v1: 핵융합")  # the same content saved again
    assert sub.deliveries == 1 and sub.delivered_this_period == 0
    net.advance(P)  # nothing new in period 2 -> refunded, not paid
    assert sub.periods_paid == 1 and sub.periods_refunded == 1
    note("v2: 새 발견")
    note("v2: 새 발견")
    assert sub.deliveries == 2
    buyer = net.nodes["n1"].memory
    live = [r.content for r in buyer.recall("구독 과학 소식 핵융합 발견", k=8) if r.source == "deal:n4"]
    assert len(live) == 1 and "v2: 새 발견" in live[0]  # the older delivery is superseded


def test_renewals_are_noted_for_the_subscriber_and_count_as_spending():
    net, sub, _ = subscribed_net()
    buyer = net.nodes["n1"]
    net.advance(P)
    assert sub.state == "ACTIVE" and sub.period_start == net.board.height
    notice = buyer.memory.records[-1]
    assert notice.source == f"chain:{sub.sub_id}" and "자동 갱신" in notice.content and "3P" in notice.content
    assert (net.board.height, 3000) in buyer.policy.spends
    assert any(e["event"] == "subscription_renewed" and e["node"] == "n1" for e in net.events)


def test_core_cancels_a_renewal_that_would_break_the_spending_limit():
    net, sub, _ = subscribed_net()
    buyer = net.nodes["n1"]
    net.advance(P - 2)
    limit = buyer.policy.hourly_limit_mP(net.board.account("n1").available_mP)
    buyer.policy.record_spend(limit - 1000, net.board.height)  # a big spend this hour
    before = net.board.account("n1").balance_mP
    net.advance(1)  # the last block of the period: Core cancels before the renewal
    assert sub.state == "ENDED" and sub.periods_paid == 1
    net.advance(5)
    # The delivered period was paid from its escrow; no new period was escrowed.
    assert net.board.account("n1").balance_mP == before and net.board.escrow_mP == 0
    assert "해지(Core)" in buyer.memory.records[-1].content
    assert any(e["event"] == "subscription_cancelled" and e["by"] == "core" for e in net.events)
    assert net.board.total_supply_check()
