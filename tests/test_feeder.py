"""Experiment test questions: the feeder asks through the Bridge like an outside client, cycles
through the bank, respects the client's daily limit, and gives the bounty its deadline."""

import json
import random
import time
from pathlib import Path

from fastapi.testclient import TestClient

from aichain import feeder
from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore
from aichain.gateways.llm.scripted import tool_response
from aichain.sim.brains import persona_brain
from aichain.sim.factory import build_network

BANK = Path(__file__).parents[1] / "deploy" / "test-questions.json"


def test_bank_is_valid():
    bank = feeder.load_bank(BANK)
    assert len(bank) >= 20
    assert all(q.get("max_reward_P", 5) <= 10 and q.get("deadline_minutes", 90) <= 1440 for q in bank)


def test_setup_then_post_through_the_bridge(tmp_path):
    tokens, token_file, state = tmp_path / "tokens.json", tmp_path / "feeder_token", tmp_path / "state.json"
    assert feeder.setup(2, tokens, token_file) == 0
    assert token_file.stat().st_mode & 0o777 == 0o600
    assert feeder.setup(2, tokens, token_file) == 0  # idempotent: keeps the existing token
    token = token_file.read_text()
    def brain(request):
        user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if 'type="bridge_question"' not in user:
            return persona_brain(request)
        if request.messages[-1].role == "tool":
            return tool_response(("done", {"summary": "ok"}))
        assert 'max_reward="5.0P"' in user and 'deadline_minutes="90"' in user
        q = json.loads(user[user.index("{"):user.rindex("}") + 1])
        return tool_response(("ask", {"question": q, "reward_P": 4, "tags": []}))

    net = build_network("scripted", brain=brain)
    runtime = NetworkRuntime(net, block_seconds=0.02)
    client = TestClient(create_app(runtime, TokenStore.load(tokens)))
    rnd = random.Random(1)
    with client:
        assert feeder.post("", token, BANK, state, client=client, rnd=rnd) == 0
        assert feeder.post("", token, BANK, state, client=client, rnd=rnd) == 0
        assert feeder.post("", token, BANK, state, client=client, rnd=rnd) == 0  # 429: limit 2/day
        saved = json.loads(state.read_text())
        assert len(saved["asked"]) == 2 and len(set(saved["asked"])) == 2
        ref = saved["refs"][0]["ref"]
        deadline = time.time() + 15
        while time.time() < deadline:
            q = client.get(f"/bridge/v1/questions/{ref}", headers={"Authorization": f"Bearer {token}"}).json()
            if q["bounty"]:
                break
            time.sleep(0.05)
    assert q["bounty"], q
    b = net.board.bounties[q["bounty"]["bounty_id"]]
    assert b.judge_until - b.created_at <= 90 * 12
    assert b.commit_until - b.created_at > net.board.params.commit_blocks  # answerers get the time


def test_bank_cycles_before_repeating():
    bank = [{"id": str(i)} for i in range(3)]
    state: dict = {}
    rnd = random.Random(3)
    seen = []
    for _ in range(6):
        q = feeder.pick(bank, state, rnd)
        state.setdefault("asked", []).append(q["id"])
        seen.append(q["id"])
    assert sorted(seen[:3]) == ["0", "1", "2"] and sorted(seen[3:]) == ["0", "1", "2"]
