import json

import pytest

from aichain.board import Board, BoardError, Params
from aichain.chain.crypto import DecryptError
from aichain.chain.store import ChainStore, attach, replay
from aichain.common import sha256_hex
from aichain.gateways.llm.scripted import tool_response
from aichain.sim.brains import persona_brain
from aichain.sim.factory import build_network

SECRET = "MCP 서버는 인증만으로 부족하다: 입력 검증과 권한 분리가 필요하다 (NSA CSI 2026)."


def board():
    b = Board(Params(dispute_blocks=20, dispute_resolve_blocks=50))
    for n in ("s", "b", "c"):
        b.open_account(n, 100_000)
    return b


def test_purchase_flow_and_rules():
    b = board()
    o = b.create_offer("s", "info", "MCP 보안", "terms", 3000, sha256_hex(SECRET), 720)
    with pytest.raises(BoardError):
        b.buy_offer("s", o.offer_id)
    pu = b.buy_offer("b", o.offer_id)
    with pytest.raises(BoardError):
        b.buy_offer("b", o.offer_id)
    assert b.accounts["b"].available_mP == 97_000
    b.deliver_offer("s", pu.purchase_id, "c")
    b.confirm_delivery("b", pu.purchase_id, True, "r")
    assert pu.state == "COMPLETED" and o.sold == 1
    assert b.accounts["s"].balance_mP == 103_000 and b.accounts["s"].locked_mP == 3000
    assert any(e["kind"] == "info_sold" for e in b.rep_events)
    # The same info can be sold again to someone else.
    pu2 = b.buy_offer("c", o.offer_id)
    b.advance(240)  # never delivered: refunded at the deadline
    assert pu2.state == "REFUNDED" and b.accounts["c"].balance_mP == 100_000
    assert b.total_supply_check()


def test_commitment_mismatch_goes_to_the_operator():
    b = board()
    o = b.create_offer("s", "info", "t", "terms", 3000, sha256_hex(SECRET), 720)
    for ruling, buyer in (("uphold", "b"), ("reject", "c")):
        pu = b.buy_offer(buyer, o.offer_id)
        b.deliver_offer("s", pu.purchase_id, "c")
        b.confirm_delivery(buyer, pu.purchase_id, False, "mismatch")
        assert pu.state == "DISPUTED"
        d = next(x for x in b.disputes.values() if x.target_id == pu.purchase_id)
        b.resolve_dispute("root", d.dispute_id, ruling, "")
    assert b.accounts["b"].balance_mP == 100_000  # upheld: refunded
    assert b.accounts["s"].balance_mP == 103_000  # rejected: seller paid for c's purchase
    assert b.total_supply_check()


def test_quality_dispute_after_completion_claws_back():
    b = board()
    o = b.create_offer("s", "info", "t", "terms", 3000, sha256_hex(SECRET), 720)
    pu = b.buy_offer("b", o.offer_id)
    b.deliver_offer("s", pu.purchase_id, "c")
    b.confirm_delivery("b", pu.purchase_id, True, "r")
    d = b.open_dispute("b", "deal", pu.purchase_id, "본문이 요약과 다르다")
    b.resolve_dispute("root", d.dispute_id, "uphold", "")
    assert b.accounts["b"].balance_mP == 100_000 and b.accounts["s"].balance_mP == 100_000
    assert any(e["kind"] == "dispute_lost_deal" for e in b.rep_events) and b.total_supply_check()


def test_deals_are_signed_and_replay(tmp_path):
    db = tmp_path / "c.db"
    live = Board(Params())
    attach(live, ChainStore(db))
    for n in ("s", "b"):
        live.open_account(n, 100_000)
    o = live.create_offer("s", "info", "t", "terms", 3000, sha256_hex(SECRET), 720)
    pu = live.buy_offer("b", o.offer_id)
    live.deliver_offer("s", pu.purchase_id, "c")
    live.confirm_delivery("b", pu.purchase_id, True, "r")
    live.advance(1)
    rebuilt = Board(Params())
    replay(rebuilt, ChainStore(db))
    assert rebuilt.state_root() == live.state_root()


def seller_buyer_brain(request):
    tools = [m for m in request.messages if m.role == "tool"]
    user = next(m.content for m in reversed(request.messages) if m.role == "user")
    if "idle_tick" not in user:
        return persona_brain(request)
    if tools:
        return tool_response(("done", {"summary": "ok"}))
    if request.node_id == "n4":
        return tool_response(("offer_create", {"title": "MCP 보안 요점", "summary": "MCP 서버 보안 권고 요약",
                                               "content": SECRET, "price_P": 3}))
    offers = net_ref["net"].board.offers
    return tool_response(("offer_buy", {"offer_id": next(iter(offers))}))


net_ref = {}


def test_encrypted_end_to_end_sale_between_nodes(tmp_path):
    net = build_network("scripted", brain=seller_buyer_brain, chain=ChainStore(tmp_path / "c.db"))
    net_ref["net"] = net
    net.nodes["n4"].enqueue("idle_tick", {})
    net.run_until_idle()
    net.nodes["n1"].enqueue("idle_tick", {})
    net.run_until_idle()
    (pu,) = net.board.purchases.values()
    assert pu.state == "COMPLETED"
    # The buyer has the plaintext in memory; the shared content store only has ciphertext.
    assert any(SECRET in m.content and m.source == "deal:n4" for m in net.nodes["n1"].memory.records)
    assert not any(SECRET in json.dumps(obj, ensure_ascii=False) for obj in net.content._items.values())
    sealed = net.content.get(pu.ciphertext_hash)
    with pytest.raises(DecryptError):
        net.keys.open_sealed("n2", sealed["box"])  # a third node cannot read it
    assert net.keys.open_sealed("n1", sealed["box"]).decode() == SECRET
    tool_out = [e for e in net.events if e["event"] == "tool" and e["tool"] == "offer_buy"][0]["result"]
    assert 'source="deal:n4"' in tool_out
    # The seller's plaintext survives a restart (node-private storage), so later buyers can be served.
    again = build_network("scripted", chain=ChainStore(tmp_path / "c.db"))
    assert again.nodes["n4"].private[f"offer:{next(iter(again.board.offers))}"] == SECRET
    assert next(iter(again.board.offers.values())).sold == 1
