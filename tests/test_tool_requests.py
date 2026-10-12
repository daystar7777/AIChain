import json
import os
import sys

import pytest

from aichain.board import Board, BoardError, Params
from aichain.chain.store import ChainStore, attach, replay
from aichain.gateways.llm.scripted import tool_response
from aichain.sim.brains import persona_brain
from aichain.sim.factory import build_network
from aichain.workshop.executor import LocalDevExecutor
from aichain.workshop.registry import Workshop

TESTS = [{"input": {"text": "a b c"}, "expected": {"words": 3}},
         {"input": {"text": ""}, "expected": {"words": 0}},
         {"input": {"text": "하나 둘"}, "expected": {"words": 2}}]
GOOD = "def run(input, ctx):\n    return {'words': len(input['text'].split())}\n"
BUGGY = "def run(input, ctx):\n    return {'words': len(input['text'])}\n"
MANIFEST = {"name": "word-count", "description": "텍스트의 단어 수를 센다",
            "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}


def board():
    b = Board(Params())
    for n in ("buyer", "s1", "s2"):
        b.open_account(n, 100_000)
    return b


def test_board_rules_and_settlement():
    b = board()
    req = b.create_tool_request("buyer", "word counter", "spec", "tests", 10_000, 240)
    assert b.accounts["buyer"].available_mP == 90_000
    b.publish_tool("s1", "s1.wc", 1, "h1", 0, 0)
    b.publish_tool("buyer", "buyer.wc", 1, "h", 0, 0)
    with pytest.raises(BoardError):
        b.submit_tool("buyer", req.request_id, "buyer.wc", 1)  # own request
    with pytest.raises(BoardError):
        b.submit_tool("s2", req.request_id, "s1.wc", 1)  # not your tool
    b.submit_tool("s1", req.request_id, "s1.wc", 1)
    b.verify_submission("root", req.request_id, "s1.wc", 1, False, "r")
    assert req.state == "OPEN"
    b.publish_tool("s1", "s1.wc", 2, "h2", 0, 0)
    b.submit_tool("s1", req.request_id, "s1.wc", 2)
    b.verify_submission("root", req.request_id, "s1.wc", 2, True, "r")
    assert req.state == "FULFILLED" and req.winner == "s1"
    assert b.accounts["s1"].balance_mP == 110_000 and b.accounts["s1"].locked_mP == 10_000
    assert b.has_license("buyer", "s1.wc", 2)
    assert any(e["kind"] == "tool_licensed" and e.get("via") == req.request_id for e in b.rep_events)
    with pytest.raises(BoardError):
        b.submit_tool("s2", req.request_id, "s1.wc", 2)
    assert b.total_supply_check()


def test_expiry_refunds_and_submission_limit():
    b = board()
    req = b.create_tool_request("buyer", "t", "s", "t", 5_000, 120)
    for v in (1, 2, 3, 4):
        b.publish_tool("s1", "s1.x", v, f"h{v}", 0, 0)
    for v in (1, 2, 3):
        b.submit_tool("s1", req.request_id, "s1.x", v)
    with pytest.raises(BoardError):
        b.submit_tool("s1", req.request_id, "s1.x", 4)
    b.advance(120)
    assert req.state == "EXPIRED" and b.accounts["buyer"].balance_mP == 100_000


def test_trades_are_signed_and_replay(tmp_path):
    db = tmp_path / "c.db"
    live = Board(Params())
    attach(live, ChainStore(db))
    for n in ("buyer", "s1"):
        live.open_account(n, 100_000)
    req = live.create_tool_request("buyer", "t", "s", "t", 5_000, 240)
    live.publish_tool("s1", "s1.x", 1, "h", 0, 0)
    live.submit_tool("s1", req.request_id, "s1.x", 1)
    live.verify_submission("root", req.request_id, "s1.x", 1, True, "r")
    live.advance(1)
    ver = next(e for e in ChainStore(db).entries() if e[0] == "tx" and e[3]["type"] == "verify_submission")
    assert ver[3]["from"] == "root"
    rebuilt = Board(Params())
    replay(rebuilt, ChainStore(db))
    assert rebuilt.state_root() == live.state_root()


@pytest.fixture
def dev_exec(monkeypatch):
    if not sys.platform.startswith("darwin") and os.environ.get("AICHAIN_DEV_EXEC_TESTS") != "1":
        pytest.skip("code-execution tests run only on a development Mac")
    monkeypatch.setenv("AICHAIN_UNSAFE_LOCAL_EXEC", "1")
    return LocalDevExecutor()


def test_buy_order_is_filled_by_another_node(tmp_path, dev_exec):
    seller_steps = [
        ("tool_dev", {"manifest": MANIFEST, "code": BUGGY, "tests": [
            {"input": {"text": ""}, "expected": {"words": 0}}, {"input": {"text": "x"}, "expected": {"words": 1}},
            {"input": {"text": "y"}, "expected": {"words": 1}}], "reason": "구매 요청 대응"}),
        ("tool_publish", {"tool_id": "n2.word-count", "version": 1, "price_install_P": 1, "price_call_P": 0}),
        ("tool_submit", {"request_id": "REQ", "tool_id": "n2.word-count", "version": 1}),
        ("tool_dev", {"manifest": MANIFEST, "code": GOOD, "tests": TESTS, "reason": "버그 수정"}),
        ("tool_publish", {"tool_id": "n2.word-count", "version": 2, "price_install_P": 1, "price_call_P": 0}),
        ("tool_submit", {"request_id": "REQ", "tool_id": "n2.word-count", "version": 2}),
    ]
    state = {}

    def brain(request):
        tools = [m for m in request.messages if m.role == "tool"]
        user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if request.node_id == "n1" and "idle_tick" in user:
            if tools:
                return tool_response(("done", {"summary": "요청함"}))
            return tool_response(("tool_request", {"title": "단어 수 세기", "description": "공백 기준 단어 수",
                                                   "tests": TESTS, "reward_P": 8, "deadline_hours": 2}))
        if "tool_request_new" in user:
            if request.node_id != "n2":
                return tool_response(("done", {"summary": "패스"}))
            import re
            state["req"] = re.search(r'request_id="([^"]+)"', user).group(1)
            if len(tools) < len(seller_steps):
                name, args = seller_steps[len(tools)]
                return tool_response((name, json.loads(json.dumps(args).replace("REQ", state["req"]))))
            return tool_response(("done", {"summary": "납품"}))
        return persona_brain(request)

    net = build_network("scripted", brain=brain, workshop=Workshop(dev_exec, tmp_path / "w.db"))
    net.nodes["n1"].enqueue("idle_tick", {})
    net.run_until_idle()
    (req,) = net.board.tool_requests.values()
    assert req.state == "FULFILLED" and req.winner_tool == ["n2.word-count", 2]
    assert [s["passed"] for s in req.submissions] == [False, True]
    assert net.board.has_license("n1", "n2.word-count", 2)
    assert net.board.accounts["n1"].balance_mP == 992_000 and net.board.accounts["n2"].locked_mP == 8_000
    trades = json.loads(net.nodes["n3"]._exec_tool("history", json.dumps({"what": "trades"})).split("\n", 1)[1].rsplit("\n", 1)[0])
    assert any(t.get("kind") == "tool_request" and t["answerer"] == "n2" and t["accepted"] for t in trades)
    assert any(m.source == f"chain:{req.request_id}" for m in net.nodes["n1"].memory.records)
