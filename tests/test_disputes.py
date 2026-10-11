import json

import pytest
from fastapi.testclient import TestClient

from aichain.board import Board, BoardError, BountyState, Params, commitment_of
from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.chain.store import ChainStore, attach, replay
from aichain.gateways.llm.scripted import tool_response
from aichain.reputation import for_board
from aichain.sim.brains import persona_brain
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS, build_network

P = 1000


def paid_bounty(board=None, reject_all=False):
    b = board or Board(Params(commit_blocks=12, reveal_blocks=6, judge_blocks=10, dispute_blocks=20,
                              dispute_resolve_blocks=50))
    for n in ("asker", "a1", "a2", "w"):
        if n not in b.accounts:
            b.open_account(n, 1000 * P)
    bounty = b.create_bounty("asker", "q", 20 * P, [])
    ans = {n: b.commit_answer(n, bounty.bounty_id, commitment_of("h", "s", n)) for n in ("a1", "a2")}
    b.advance(12)
    for n, a in ans.items():
        b.reveal_answer(n, a.answer_id, "h", "s")
    b.advance(6)
    if reject_all:
        for a in ans.values():
            b.reject_answer("asker", bounty.bounty_id, a.answer_id, "wrong")
        b.advance(10)
    else:
        b.accept_answer("asker", bounty.bounty_id, ans["a1"].answer_id)
    return b, bounty, ans


def test_upheld_answer_dispute_claws_back_with_finder_reward():
    b, bounty, ans = paid_bounty()
    d = b.open_dispute("w", "answer", ans["a1"].answer_id, "claim")
    assert d.bond_mP == 2 * P and b.accounts["w"].balance_mP == 998 * P  # 10% of 20P
    b.resolve_dispute("root", d.dispute_id, "uphold", "note")
    assert b.accounts["a1"].balance_mP == 1000 * P  # payout clawed back (bond returned earlier)
    assert b.accounts["w"].balance_mP == 1004 * P   # bond back + 20% of 20P
    assert b.accounts["asker"].balance_mP == 996 * P  # 80% of the reward returned
    assert b.total_supply_check()
    kinds = {(e["account"], e["kind"]) for e in b.rep_events}
    assert ("a1", "dispute_lost_answer") in kinds and ("w", "dispute_won") in kinds


def test_rejected_dispute_splits_the_bond():
    b, bounty, ans = paid_bounty()
    d = b.open_dispute("w", "answer", ans["a1"].answer_id, "weak claim")
    b.resolve_dispute("root", d.dispute_id, "reject", "no evidence")
    assert b.accounts["w"].balance_mP == 998 * P
    assert b.accounts["a1"].balance_mP == 1021 * P  # payout kept + half the bond
    assert b.burned_mP == 1 * P and b.total_supply_check()
    assert for_board(b, "w").terms_e6["W"] < 500_000


def test_dispute_without_ruling_is_voided_at_the_deadline():
    b, bounty, ans = paid_bounty()
    d = b.open_dispute("w", "answer", ans["a1"].answer_id, "c")
    b.advance(50)
    assert d.state == "void" and b.accounts["w"].balance_mP == 1000 * P


def test_rejection_dispute_on_refunded_bounty():
    b, bounty, ans = paid_bounty(reject_all=True)
    assert bounty.state == BountyState.REFUNDED
    with pytest.raises(BoardError):
        b.open_dispute("w", "rejection", ans["a1"].answer_id, "c")  # only the author
    d = b.open_dispute("a1", "rejection", ans["a1"].answer_id, "met the criteria")
    b.resolve_dispute("root", d.dispute_id, "uphold", "")
    assert b.accounts["a1"].balance_mP == 1010 * P  # reward / 2 revealed answers
    kinds = [e["kind"] for e in b.rep_events if e["account"] == "a1"]
    assert "answer_rejected_overturned" in kinds and "answer_rejected" not in kinds
    assert b.total_supply_check()


def test_dispute_rules():
    b, bounty, ans = paid_bounty()
    with pytest.raises(BoardError):
        b.open_dispute("asker", "answer", ans["a1"].answer_id, "c")  # asker cannot
    with pytest.raises(BoardError):
        b.open_dispute("w", "answer", ans["a2"].answer_id, "c")  # a2 was not paid
    b.open_dispute("w", "answer", ans["a1"].answer_id, "c")
    with pytest.raises(BoardError):
        b.open_dispute("a2", "answer", ans["a1"].answer_id, "c")  # already open


def test_ratings_feed_the_social_term_and_can_be_disputed():
    b, bounty, ans = paid_bounty()
    base = for_board(b, "a1").terms_e6["S"]
    r = b.rate("asker", "a1", 2, bounty.bounty_id)
    assert for_board(b, "a1").terms_e6["S"] > base
    with pytest.raises(BoardError):
        b.rate("asker", "a1", 1, bounty.bounty_id)  # once per context
    with pytest.raises(BoardError):
        b.rate("w", "a1", -2, bounty.bounty_id)  # w did not take part
    bad = b.rate("a2", "a1", -2, bounty.bounty_id)
    d = b.open_dispute("a1", "rating", bad.rating_id, "baseless")
    b.resolve_dispute("root", d.dispute_id, "uphold", "")
    assert b.ratings[bad.rating_id].void
    assert for_board(b, "a1").terms_e6["S"] == 1_000_000  # only the +2 counts now


def test_rulings_are_root_signed_and_replay(tmp_path):
    db = tmp_path / "c.db"
    live = Board(Params(commit_blocks=12, reveal_blocks=6, judge_blocks=10, dispute_blocks=20))
    attach(live, ChainStore(db))
    _, bounty, ans = paid_bounty(live)
    d = live.open_dispute("w", "answer", ans["a1"].answer_id, "c")
    live.resolve_dispute("root", d.dispute_id, "uphold", "n")
    live.advance(1)
    entries = ChainStore(db).entries()
    ruling = next(e for e in entries if e[0] == "tx" and e[3]["type"] == "resolve_dispute")
    assert ruling[3]["from"] == "root"
    rebuilt = Board(live.params)
    replay(rebuilt, ChainStore(db))
    assert rebuilt.state_root() == live.state_root()


def test_warden_disputes_and_operator_rules_through_the_bridge(tmp_path):
    def brain(request):
        tools = [m for m in request.messages if m.role == "tool"]
        last_user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if request.node_id == "n3" and "review_settled" in last_user:
            if tools:
                return tool_response(("done", {"summary": "분쟁 제기"}))
            import re
            aid = re.search(r'<answer id="([^"]+)" author="n2" verdict="수락"', last_user).group(1)
            return tool_response(("dispute", {"target_kind": "answer", "target_id": aid,
                                              "claim": "정답 조건(실행 예시) 미충족", "evidence": [aid]}))
        return persona_brain(request)

    net = build_network("scripted", brain=brain)
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    p = SHORT_PARAMS
    net.advance(p.commit_blocks + p.reveal_blocks + p.judge_blocks)
    (d,) = net.board.disputes.values()
    assert (d.opener, d.defendant, d.state) == ("n3", "n2", "open")

    path = tmp_path / "t.json"
    add_client(path, "c", 5)
    client = TestClient(create_app(NetworkRuntime(net), TokenStore.load(path), start_runtime=False,
                                   admin_token="adm_secret"))
    assert client.get("/admin/v1/disputes").status_code == 401
    hdr = {"Authorization": "Bearer adm_secret"}
    (item,) = client.get("/admin/v1/disputes", headers=hdr).json()
    assert item["claim"]["claim"].startswith("정답 조건") and item["target"]["author"] == "n2"
    r = client.post(f"/admin/v1/disputes/{d.dispute_id}/resolve", headers=hdr,
                    json={"ruling": "reject", "note": "예시 코드 있음"})
    assert r.status_code == 200 and r.json()["state"] == "rejected"
    assert client.post(f"/admin/v1/disputes/{d.dispute_id}/resolve", headers=hdr,
                       json={"ruling": "uphold"}).status_code == 409
    assert net.board.total_supply_check()
    assert any(m.source == f"chain:{d.dispute_id}" for m in net.nodes["n2"].memory.records)
