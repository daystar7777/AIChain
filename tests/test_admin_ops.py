"""Operator commands from the 2026-10-12 gap review: closing an implemented proposal, blocking a
tool, and re-announcing open proposals to nodes that have not voted."""

from fastapi.testclient import TestClient

from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.reputation import EPOCH_BLOCKS
from aichain.sim.factory import build_network

P = 1000


def client_for(net, tmp_path):
    path = tmp_path / "t.json"
    add_client(path, "c", 5)
    app = create_app(NetworkRuntime(net), TokenStore.load(path), start_runtime=False, admin_token="adm")
    return TestClient(app), {"Authorization": "Bearer adm"}


def test_operator_closes_an_implementing_proposal(tmp_path):
    net = build_network("scripted")
    b = net.board
    b.params.vote_blocks, b.params.timelock_blocks = 20, 10
    prop = b.create_proposal("n1", "upgrade", "t", net.put_content({"spec": "s"}, author="n1"), [])
    for n in ("n2", "n3", "n4"):
        b.vote(n, prop.proposal_id, "yes", "r")
    b.advance(31)
    assert prop.state == "IMPLEMENTING"
    client, hdr = client_for(net, tmp_path)
    r = client.post(f"/admin/v1/proposals/{prop.proposal_id}/close", headers=hdr,
                    json={"outcome": "nope", "note": "x"})
    assert r.status_code == 422
    r = client.post(f"/admin/v1/proposals/{prop.proposal_id}/close", headers=hdr,
                    json={"outcome": "implemented", "note": "배포 완료"})
    assert r.status_code == 200 and prop.state == "IMPLEMENTED" and prop.closed_hash
    r = client.post(f"/admin/v1/proposals/{prop.proposal_id}/close", headers=hdr,
                    json={"outcome": "abandoned", "note": "again"})
    assert r.status_code == 409


def test_operator_blocks_a_tool(tmp_path):
    net = build_network("scripted")
    b = net.board
    b.publish_tool("n1", "n1.t", 1, "code", 1 * P, 100)
    b.license_tool("n2", "n1.t", 1)
    client, hdr = client_for(net, tmp_path)
    r = client.post("/admin/v1/tools/n1.t/block", headers=hdr, json={"version": 1, "reason": "악성 코드"})
    assert r.status_code == 200 and r.json()["blocked"]
    assert not b.has_license("n2", "n1.t", 1)
    r = client.post("/admin/v1/tools/n1.t/block", headers=hdr, json={"version": 2, "reason": "없는 버전"})
    assert r.status_code == 409


def test_open_proposal_is_reannounced_to_nodes_that_have_not_voted(monkeypatch):
    net = build_network("scripted")
    b = net.board
    prop = b.create_proposal("n1", "upgrade", "t", net.put_content({"spec": "s"}, author="n1"), [])
    b.vote("n2", prop.proposal_id, "yes", "r")
    for node in net.nodes.values():
        node._queue.clear()
    net.remind_voters()
    queued = {n.id for n in net.nodes.values()
              if any(ev.kind == "proposal_open" for ev in n._queue)}
    assert queued == {"n3", "n4"}
    net.remind_voters()
    assert sum(ev.kind == "proposal_open" for ev in net.nodes["n3"]._queue) == 1
    # advance() runs the reminder on the hour (episodes held back so the queue can be checked)
    monkeypatch.setattr(net, "run_until_idle", lambda *a, **k: True)
    net.nodes["n3"]._queue.clear()
    net.advance(EPOCH_BLOCKS - b.height % EPOCH_BLOCKS)
    assert any(ev.kind == "proposal_open" for ev in net.nodes["n3"]._queue)


def test_client_deadline_and_min_reputation_reach_the_bounty():
    from aichain.gateways.llm.scripted import tool_response
    from aichain.sim.brains import persona_brain
    from aichain.sim.demo import DEFAULT_QUESTION

    seen = {}

    def brain(request):
        user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if request.node_id == "n1" and "bridge_question" in user:
            if request.messages[-1].role == "tool":
                return tool_response(("done", {"summary": "ok"}))
            return tool_response(("ask", {"question": DEFAULT_QUESTION, "reward_P": 5, "tags": [],
                                          "min_reputation": 0.2}))
        if "bounty_new" in user and "min_reputation=" in user:
            seen["shown"] = True
        return persona_brain(request)

    net = build_network("scripted", brain=brain)
    net.bridge_question("n1", "alice", DEFAULT_QUESTION["title"], DEFAULT_QUESTION["body"],
                        DEFAULT_QUESTION["acceptance_criteria"], deadline_minutes=5)
    net.run_until_idle()
    (b,) = net.board.bounties.values()
    assert b.min_rep_e6 == 200_000
    assert b.judge_until - b.created_at <= 5 * 12
    assert seen.get("shown")


def test_one_node_failing_does_not_stop_the_others(monkeypatch):
    from aichain.board import commitment_of

    net = build_network("scripted")
    b = net.board
    bounty = b.create_bounty("n1", "q", 5 * P, [])
    for n in ("n2", "n3"):
        body = {"answer": f"from {n}", "confidence": 0.5, "sources": [], "verification": ""}
        salt = "s" * 32
        ans = b.commit_answer(n, bounty.bounty_id, commitment_of(net.content.hash_of(body), salt, n))
        net.nodes[n]._sealed[ans.answer_id] = (body, salt)
    monkeypatch.setattr(net, "run_until_idle", lambda *a, **k: True)

    def broken():
        raise RuntimeError("memory store unavailable")
    monkeypatch.setattr(net.nodes["n2"], "run_duties", broken)
    monkeypatch.setattr(net.nodes["n2"], "on_settled", lambda payload: broken())
    net.advance(b.params.commit_blocks + 1)
    assert [b.answers[a].revealed for a in bounty.answers] == [False, True]  # n3 still revealed
    net.advance(b.params.reveal_blocks + b.params.judge_blocks + 1)
    assert bounty.state.name == "AUTO_SPLIT"
    errors = [e for e in net.events if e["event"] == "node_error"]
    assert {e["step"] for e in errors} >= {"run_duties", "on_settled"}
    assert any(m.source == f"chain:{bounty.bounty_id}" for m in net.nodes["n3"].memory.records)
