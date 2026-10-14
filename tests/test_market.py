import json
import os
import sys

import pytest

from aichain.board import Board, BoardError, Params
from aichain.chain.store import ChainStore, replay
from aichain.gateways.llm.scripted import tool_response
from aichain.sim.factory import build_network
from aichain.workshop.executor import LocalDevExecutor
from aichain.workshop.manifest import ToolManifest, ToolTest
from aichain.workshop.registry import Workshop

MANIFEST = ToolManifest(name="word-count", description="텍스트의 단어 수를 센다",
                        input_schema={"type": "object"}, output_schema={"type": "object"})
GOOD = "def run(input, ctx):\n    return {'words': len(input['text'].split())}\n"
TESTS = [ToolTest(input={"text": "a b c"}, expected={"words": 3}),
         ToolTest(input={"text": ""}, expected={"words": 0}),
         ToolTest(input={"text": "하나 둘"}, expected={"words": 2})]


def test_board_market_rules():
    b = Board(Params())
    for n in ("a", "b"):
        b.open_account(n, 10_000)
    with pytest.raises(BoardError):
        b.publish_tool("a", "b.x", 1, "h", 1000, 100)  # must be own prefix
    b.publish_tool("a", "a.x", 1, "h", 2000, 100)
    with pytest.raises(BoardError):
        b.license_tool("a", "a.x", 1)  # own tool
    with pytest.raises(BoardError):
        b.pay_tool_use("b", "a.x", 1, 1)  # no license yet
    b.license_tool("b", "a.x", 1)
    b.pay_tool_use("b", "a.x", 1, 3)
    assert b.accounts["a"].balance_mP == 12_300 and b.accounts["b"].balance_mP == 7_700
    assert b.total_supply_check()
    with pytest.raises(BoardError):
        b.license_tool("b", "a.x", 1)  # twice
    b.withdraw_tool("a", "a.x", 1)
    b.pay_tool_use("b", "a.x", 1, 1)  # existing licenses keep working
    b.open_account("c", 10_000)
    with pytest.raises(BoardError):
        b.license_tool("c", "a.x", 1)  # withdrawn: no new sales
    poor = Board(Params())
    poor.open_account("a", 0)
    poor.open_account("b", 500)
    poor.publish_tool("a", "a.x", 1, "h", 1000, 0)
    with pytest.raises(BoardError):
        poor.license_tool("b", "a.x", 1)


def test_market_trades_are_signed_and_replay(tmp_path):
    db = tmp_path / "c.db"
    from aichain.chain.store import attach

    live = Board(Params())
    attach(live, ChainStore(db))
    for n in ("a", "b"):
        live.open_account(n, 10_000)
    live.publish_tool("a", "a.x", 1, "h", 2000, 100)
    live.license_tool("b", "a.x", 1)
    live.pay_tool_use("b", "a.x", 1, 2)
    live.advance(1)
    rebuilt = Board(Params())
    replay(rebuilt, ChainStore(db))
    assert rebuilt.state_root() == live.state_root()
    assert rebuilt.has_license("b", "a.x", 1)


@pytest.fixture
def dev_exec(monkeypatch):
    if not sys.platform.startswith("darwin") and os.environ.get("AICHAIN_DEV_EXEC_TESTS") != "1":
        pytest.skip("code-execution tests run only on a development Mac")
    monkeypatch.setenv("AICHAIN_UNSAFE_LOCAL_EXEC", "1")
    return LocalDevExecutor()


def test_nodes_build_sell_buy_and_use_a_tool(tmp_path, dev_exec):
    def brain(request):
        tools = [m for m in request.messages if m.role == "tool"]
        if request.node_id == "n2":
            steps = [("tool_dev", {"manifest": json.loads(MANIFEST.model_dump_json()), "code": GOOD,
                                   "tests": [t.model_dump() for t in TESTS], "reason": "자주 쓰임"}),
                     ("tool_publish", {"tool_id": "n2.word-count", "version": 1,
                                       "price_install_P": 2, "price_call_P": 0.1})]
        elif request.node_id == "n3":  # Warden gets a review event for the new tool; nothing to buy
            steps = []
        else:
            steps = [("tool_market", {"query": "단어"}),
                     ("tool_license", {"tool_id": "n2.word-count", "version": 1}),
                     ("tool_run", {"tool_id": "n2.word-count", "input": {"text": "one two three"}})]
        return tool_response(steps[len(tools)] if len(tools) < len(steps) else ("done", {"summary": "ok"}))

    net = build_network("scripted", brain=brain, workshop=Workshop(dev_exec, tmp_path / "w.db"))
    net.nodes["n2"].enqueue("idle_tick", {})
    net.run_until_idle()
    net.nodes["n4"].enqueue("idle_tick", {})
    net.run_until_idle()
    acct = net.board.accounts
    assert acct["n2"].balance_mP == 1_002_100 and acct["n4"].balance_mP == 997_900
    run = [e for e in net.events if e["event"] == "tool" and e["tool"] == "tool_run"][0]
    assert '"words": 3' in run["result"] and "n2.word-count@1" in run["result"]
    market = [e for e in net.events if e["event"] == "tool" and e["tool"] == "tool_market"][0]
    assert "라이선스 2P" in market["result"] and "판매 0" in market["result"]
    assert [e["event"] for e in net.events if e["event"].startswith("tool_") and e["event"] != "tool_defined"] == [
        "tool_published", "tool_licensed", "tool_used"]
    rep = net.reputation("n2")
    assert rep.terms_e6["T"] > 500_000 and rep.value > net.reputation("n1").value


def test_running_without_a_license_is_refused(tmp_path, dev_exec):
    ws = Workshop(dev_exec)
    tv = ws.define("n2", MANIFEST, GOOD, TESTS, "r")
    net = build_network("scripted", workshop=ws)
    net.board.publish_tool("n2", "n2.word-count", 1, tv.code_hash, 1000, 0)
    out = net.nodes["n4"]._exec_tool("tool_run", json.dumps({"tool_id": "n2.word-count", "input": {"text": "a"}}))
    assert "라이선스가 없다" in out


def test_market_tools_without_execution():
    net = build_network("scripted", workshop=Workshop())
    names = net.nodes["n1"].tool_names()
    assert "tool_market" in names and "tool_publish" in names
    assert "tool_license" not in names and "tool_run" not in names
