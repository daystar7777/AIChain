import json
import time

import pytest
from fastapi.testclient import TestClient

from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.gateways.llm.scripted import tool_response
from aichain.sim.brains import persona_brain
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import build_network


@pytest.fixture
def bridge(tmp_path):
    def make(brain=persona_brain, daily_limit=20):
        path = tmp_path / "tokens.json"
        alice = add_client(path, "alice", daily_limit)
        bob = add_client(path, "bob", daily_limit)
        net = build_network("scripted", brain=brain)
        runtime = NetworkRuntime(net, block_seconds=0.001)
        app = create_app(runtime, TokenStore.load(path), sse_poll_seconds=0.01)
        return TestClient(app), {"alice": alice, "bob": bob}, runtime
    return make


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def wait_for(client, ref, token, status, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        q = client.get(f"/bridge/v1/questions/{ref}", headers=auth(token)).json()
        if q["status"] == status:
            return q
        time.sleep(0.02)
    raise AssertionError(f"status stayed {q['status']}")


def test_question_runs_to_settlement_with_events(bridge):
    client, tokens, runtime = bridge()
    with client:
        r = client.post("/bridge/v1/questions", headers=auth(tokens["alice"]),
                        json={**DEFAULT_QUESTION, "max_reward_P": 30})
        assert r.status_code == 202
        ref = r.json()["ref"]
        assert r.json()["links"]["events"] == f"/bridge/v1/questions/{ref}/events"

        with client.stream("GET", f"/bridge/v1/questions/{ref}/events",
                           headers=auth(tokens["alice"])) as s:
            assert s.headers["content-type"].startswith("text/event-stream")
            body = "".join(s.iter_text())
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        kinds = [e["event"] for e in events]
        assert kinds[:3] == ["queued", "processing", "bounty_open"]
        assert kinds.count("answer_committed") == 2
        assert kinds[-3:] == ["revealing", "judging", "settled"]
        assert [e["id"] for e in events] == list(range(1, len(events) + 1))

        q = client.get(f"/bridge/v1/questions/{ref}", headers=auth(tokens["alice"])).json()
        assert q["result"]["outcome"] == "PAID"
        accepted = [a for a in q["result"]["answers"] if a["accepted"]]
        assert accepted[0]["author"] == "n2" and "lst[::-1]" in accepted[0]["body"]["answer"]

        # Resume after the 3rd event with Last-Event-ID.
        with client.stream("GET", f"/bridge/v1/questions/{ref}/events",
                           headers={**auth(tokens["alice"]), "Last-Event-ID": "3"}) as s:
            resumed = [json.loads(line[6:]) for line in "".join(s.iter_text()).splitlines()
                       if line.startswith("data: ")]
        assert resumed[0]["id"] == 4

        net = client.get("/bridge/v1/network", headers=auth(tokens["bob"])).json()
        assert [n["id"] for n in net["nodes"]] == ["n1", "n2", "n3", "n4"]
        assert net["recent_bounties"][0]["bounty_id"] == q["bounty"]["bounty_id"]
    assert runtime.errors == []


def test_auth_and_isolation(bridge):
    client, tokens, _ = bridge()
    with client:
        assert client.post("/bridge/v1/questions", json=DEFAULT_QUESTION).status_code == 401
        assert client.get("/bridge/v1/network", headers=auth("nope")).status_code == 401
        ref = client.post("/bridge/v1/questions", headers=auth(tokens["alice"]),
                          json=DEFAULT_QUESTION).json()["ref"]
        assert client.get(f"/bridge/v1/questions/{ref}", headers=auth(tokens["bob"])).status_code == 404
        assert client.get(f"/bridge/v1/questions/{ref}/events",
                          headers=auth(tokens["bob"])).status_code == 404
        assert client.get("/bridge/v1/questions/q_missing",
                          headers=auth(tokens["alice"])).status_code == 404


def test_validation_and_daily_limit(bridge):
    client, tokens, _ = bridge(daily_limit=1)
    with client:
        bad = client.post("/bridge/v1/questions", headers=auth(tokens["alice"]),
                          json={"title": "x", "body": "y"})
        assert bad.status_code == 422
        assert client.post("/bridge/v1/questions", headers=auth(tokens["alice"]),
                           json=DEFAULT_QUESTION).status_code == 202
        assert client.post("/bridge/v1/questions", headers=auth(tokens["alice"]),
                           json=DEFAULT_QUESTION).status_code == 429


def test_node_may_decline_and_max_reward_is_enforced(bridge):
    def stingy_or_greedy(request):
        if request.messages[-1].role == "tool":
            return tool_response(("done", {"summary": "현상금이 상한을 넘어 거부됨"}))
        return tool_response(("ask", {"question": DEFAULT_QUESTION, "reward_P": 50, "tags": []}))

    client, tokens, runtime = bridge(brain=stingy_or_greedy)
    with client:
        ref = client.post("/bridge/v1/questions", headers=auth(tokens["alice"]),
                          json={**DEFAULT_QUESTION, "max_reward_P": 5}).json()["ref"]
        q = wait_for(client, ref, tokens["alice"], "no_bounty")
        assert q["bounty"] is None and "상한" in q["note"]
    assert not runtime.net.board.bounties
