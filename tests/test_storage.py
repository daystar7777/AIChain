import json

import pytest

from aichain.board import Board, BoardError, Params
from aichain.node import STORAGE_BASE_KB
from aichain.sim.factory import build_network


def test_lease_moves_quota_for_its_duration():
    net = build_network("scripted")
    seller, buyer = net.nodes["n2"], net.nodes["n1"]
    out = seller._exec_tool("offer_create", json.dumps({"kind": "storage", "quantity": 200, "title": "200KB",
                                                        "summary": "남는 공간", "price_P": 1, "period_hours": 1}))
    assert "저장 공간 임대 등록" in out
    offer = next(iter(net.board.offers.values()))
    out = buyer._exec_tool("offer_buy", json.dumps({"offer_id": offer.offer_id}))
    assert "임대 완료" in out
    assert buyer.storage_quota_kb() == STORAGE_BASE_KB + 200
    assert seller.storage_quota_kb() == STORAGE_BASE_KB - 200
    net.advance(720)
    assert buyer.storage_quota_kb() == STORAGE_BASE_KB == seller.storage_quota_kb()


def test_full_storage_refuses_mastermind_writes_but_not_core():
    net = build_network("scripted")
    n1 = net.nodes["n1"]
    n1.memory.remember(0, "fact", "self", 1.0, "x" * (STORAGE_BASE_KB * 1024))
    out = n1._exec_tool("remember", json.dumps({"kind": "fact", "content": "one more"}))
    assert "저장 공간이 가득" in out
    before = len(n1.memory.records)  # Core writes (episodes, chain memories) are not limited
    n1.memory.remember(0, "episode", "chain:x", 1.0, "core write")
    assert len(n1.memory.records) == before + 1


def test_cannot_lease_more_than_free_space():
    net = build_network("scripted")
    out = net.nodes["n2"]._exec_tool("offer_create", json.dumps({
        "kind": "storage", "quantity": STORAGE_BASE_KB + 1, "title": "t", "summary": "s", "price_P": 1}))
    assert "남는 저장 공간" in out


def test_board_storage_lease_bounds():
    b = Board(Params())
    b.open_account("s", 0)
    with pytest.raises(BoardError):
        b.create_offer("s", "storage", "t", "h", 1000, None, 720, quantity=20_000, period_blocks=720)


def test_lease_and_subscription_survive_a_restart(tmp_path):
    from aichain.chain.store import ChainStore

    db = tmp_path / "chain.db"
    net = build_network("scripted", chain=ChainStore(db))
    seller, buyer = net.nodes["n2"], net.nodes["n1"]
    seller._exec_tool("offer_create", json.dumps({"kind": "storage", "quantity": 200, "title": "200KB",
                                                  "summary": "남는 공간", "price_P": 1, "period_hours": 1}))
    offer = next(iter(net.board.offers.values()))
    buyer._exec_tool("offer_buy", json.dumps({"offer_id": offer.offer_id}))
    sub_offer = net.board.create_offer("n3", "subscription", "s", "terms", 1000, None, 2000, period_blocks=720)
    sub = net.board.subscribe("n4", sub_offer.offer_id)
    net.advance(100)

    again = build_network("scripted", chain=ChainStore(db))
    assert again.board.state_root() == net.board.state_root()
    assert again.nodes["n1"].storage_quota_kb() == STORAGE_BASE_KB + 200
    assert again.nodes["n2"].storage_quota_kb() == STORAGE_BASE_KB - 200
    assert again.board.subscriptions[sub.sub_id].state == "ACTIVE"
    again.advance(720)
    assert again.nodes["n1"].storage_quota_kb() == STORAGE_BASE_KB
    assert again.board.subscriptions[sub.sub_id].periods_refunded == 1  # nothing delivered
