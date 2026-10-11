import pytest
from fastapi.testclient import TestClient

from aichain.board import Board, BoardError, Params
from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.chain.store import ChainStore, attach, replay
from aichain.gateways.llm.scripted import tool_response
from aichain.sim.brains import persona_brain
from aichain.sim.factory import build_network

GOV = dict(vote_blocks=20, timelock_blocks=10, proposal_deposit_mP=50_000)


def board():
    b = Board(Params(**GOV))
    for n in ("n1", "n2", "n3", "n4"):
        b.open_account(n, 1_000_000)
    b.open_account("operator", 0)
    return b


def vote_all(b, pid, choices):
    for voter, choice in choices.items():
        b.vote(voter, pid, choice, "reason")


def test_param_proposal_passes_timelock_then_applies():
    b = board()
    prop = b.create_proposal("n1", "param", "longer judging", "h", [{"key": "judge_blocks", "to": 480}])
    assert prop.changes == [{"key": "judge_blocks", "from": 240, "to": 480}]
    assert b.accounts["n1"].available_mP == 950_000
    vote_all(b, prop.proposal_id, {"n2": "yes", "n3": "yes", "operator": "no"})
    b.advance(20)
    assert prop.state == "TIMELOCK" and prop.tally == {"yes": 2, "no": 1, "abstain": 0, "seats": 5}
    assert b.params.judge_blocks == 240
    b.advance(10)
    assert prop.state == "ACTIVATED" and b.params.judge_blocks == 480
    assert b.accounts["n1"].available_mP == 1_000_000  # deposit back
    new = b.create_bounty("n2", "q", 5000, [])
    assert new.judge_until - new.reveal_until == 480


def test_quorum_rejection_burn_and_veto():
    b = board()
    p1 = b.create_proposal("n1", "param", "t", "h", [{"key": "commit_bond_mP", "to": 1000}])
    vote_all(b, p1.proposal_id, {"n2": "yes", "n3": "yes"})  # 2 of 5: no quorum
    b.advance(20)
    assert p1.state == "EXPIRED" and b.accounts["n1"].balance_mP == 1_000_000
    p2 = b.create_proposal("n1", "param", "t", "h", [{"key": "commit_bond_mP", "to": 1000}])
    vote_all(b, p2.proposal_id, {"n2": "no", "n3": "no", "n4": "no"})
    b.advance(20)
    assert p2.state == "REJECTED" and b.accounts["n1"].balance_mP == 950_000  # >= 2/3 against: burned
    p3 = b.create_proposal("n2", "param", "t", "h", [{"key": "commit_bond_mP", "to": 1000}])
    vote_all(b, p3.proposal_id, {"n1": "yes", "n3": "yes", "n4": "yes"})
    b.advance(20)
    b.veto_proposal("root", p3.proposal_id, "unsafe")
    b.advance(10)
    assert p3.state == "VETOED" and b.params.commit_bond_mP == 500
    assert b.total_supply_check()


def test_constitution_and_rules():
    b = board()
    for bad in ([{"key": "operator_seat", "to": False}], [{"key": "unknown", "to": 1}],
                [{"key": "judge_blocks", "to": 1}]):
        with pytest.raises(BoardError):
            b.create_proposal("n1", "param", "t", "h", bad)
    with pytest.raises(BoardError):
        b.create_proposal("root", "param", "t", "h", [{"key": "judge_blocks", "to": 480}])
    p = b.create_proposal("n1", "upgrade", "new feature", "h", [])
    with pytest.raises(BoardError):
        b.create_proposal("n1", "param", "t", "h", [{"key": "judge_blocks", "to": 480}])  # one open
    with pytest.raises(BoardError):
        b.vote("n2", p.proposal_id, "yes", "")  # reason required
    vote_all(b, p.proposal_id, {"n2": "yes", "n3": "yes", "n4": "no", "operator": "abstain"})
    b.vote("n4", p.proposal_id, "yes", "changed my mind")  # last vote counts
    b.advance(30)
    assert p.state == "IMPLEMENTING"  # upgrades are implemented by people


def test_seat_proposal_removes_the_operator_seat():
    b = board()
    p = b.create_proposal("n1", "seat", "LLM only", "h", [{"key": "operator_seat", "to": False}])
    vote_all(b, p.proposal_id, {"n1": "yes", "n2": "yes", "n3": "yes", "operator": "no"})
    b.advance(30)
    assert p.state == "ACTIVATED" and "operator" not in b.voters()


def test_governance_is_signed_and_replays(tmp_path):
    db = tmp_path / "c.db"
    live = Board(Params(**GOV))
    attach(live, ChainStore(db))
    for n in ("n1", "n2", "n3", "operator"):
        live.open_account(n, 1_000_000 if n != "operator" else 0)
    p = live.create_proposal("n1", "param", "t", "h", [{"key": "judge_blocks", "to": 480}])
    vote_all(live, p.proposal_id, {"n2": "yes", "n3": "yes", "operator": "yes"})
    live.advance(30)
    rebuilt = Board(Params(**GOV))
    replay(rebuilt, ChainStore(db))
    assert rebuilt.params.judge_blocks == 480 and rebuilt.state_root() == live.state_root()


def test_nodes_vote_and_operator_votes_through_the_bridge(tmp_path):
    def brain(request):
        tools = [m for m in request.messages if m.role == "tool"]
        user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if tools:
            return tool_response(("done", {"summary": "ok"}))
        if request.node_id == "n1" and "idle_tick" in user:
            return tool_response(("propose", {"kind": "param", "title": "판정 기간 늘리기",
                                              "spec": "판정 마감이 짧아 자동 분배가 잦다.",
                                              "changes": [{"key": "judge_blocks", "to": 480}]}))
        if "proposal_open" in user:
            assert "judge_blocks: 48 → 480" in user and '<data source="proposal:' in user
            choice = "no" if request.node_id == "n3" else "yes"
            import re
            pid = re.search(r'proposal_id="([^"]+)"', user).group(1)
            return tool_response(("vote", {"proposal_id": pid, "choice": choice,
                                           "reason": "자동 분배가 줄어 판정 품질이 오른다" if choice == "yes" else "판정이 늦어지면 답변자 수입이 지연될 우려가 있다"}))
        return persona_brain(request)

    net = build_network("scripted", brain=brain)
    net.board.params.vote_blocks, net.board.params.timelock_blocks = 20, 10
    net.nodes["n1"].enqueue("idle_tick", {})
    net.run_until_idle()
    (prop,) = net.board.proposals.values()
    assert prop.votes == {"n2": "yes", "n3": "no", "n4": "yes"}

    path = tmp_path / "t.json"
    add_client(path, "c", 5)
    client = TestClient(create_app(NetworkRuntime(net), TokenStore.load(path), start_runtime=False,
                                   admin_token="adm"))
    hdr = {"Authorization": "Bearer adm"}
    (item,) = client.get("/admin/v1/proposals", headers=hdr).json()
    assert item["spec"]["title"] == "판정 기간 늘리기" and "operator" in item["voters"]
    r = client.post(f"/admin/v1/proposals/{prop.proposal_id}/vote", headers=hdr,
                    json={"choice": "yes", "reason": "운영자도 동의함"})
    assert r.status_code == 200 and r.json()["votes"]["operator"] == "yes"
    assert client.post(f"/admin/v1/proposals/{prop.proposal_id}/veto", headers=hdr,
                       json={"reason": "too early"}).status_code == 409  # not in timelock yet
    net.advance(30)
    assert prop.state == "ACTIVATED" and net.board.params.judge_blocks == 480
    assert any(m.source == f"chain:{prop.proposal_id}" for m in net.nodes["n4"].memory.records)
