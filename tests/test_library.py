import json
import os
import sys

import pytest

from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS, build_network
from aichain.workshop import library, static_check
from aichain.workshop.executor import LocalDevExecutor
from aichain.workshop.registry import Workshop


def test_every_library_tool_passes_static_checks_and_has_tests():
    names = set()
    for entry in library.CATALOG:
        code = (library.TOOLS_DIR / entry["file"]).read_text()
        assert static_check.check(code) == [], entry["name"]
        assert len(entry["tests"]) >= 3 and entry["name"] not in names
        names.add(entry["name"])
    assert len(names) >= 20


@pytest.fixture
def dev_exec(monkeypatch):
    if not sys.platform.startswith("darwin") and os.environ.get("AICHAIN_DEV_EXEC_TESTS") != "1":
        pytest.skip("code-execution tests run only on a development Mac")
    monkeypatch.setenv("AICHAIN_UNSAFE_LOCAL_EXEC", "1")
    return LocalDevExecutor()


def test_library_installs_publishes_free_and_is_idempotent(tmp_path, dev_exec):
    ws = Workshop(dev_exec, tmp_path / "w.db")
    net = build_network("scripted", workshop=ws)
    rep = net.library_report
    assert rep["failed"] == [] and len(rep["installed"]) == len(library.CATALOG)
    lib_tools = [t for t in net.board.tools.values() if t.author == "lib"]
    assert len(lib_tools) == len(library.CATALOG)
    assert all(t.price_install_mP == 0 and t.price_call_mP == 0 for t in lib_tools)
    again = library.install(Workshop(dev_exec, tmp_path / "w.db"), net.board)
    assert again["installed"] == [] and len(again["unchanged"]) == len(library.CATALOG)


def test_nodes_use_library_tools_without_license(tmp_path, dev_exec):
    calls = [("history", {"what": "trades"}),
             ("tool_run", {"tool_id": "lib.calc", "input": {"expression": "2**10"}}),
             ("tool_run", {"tool_id": "lib.text-similarity", "input": {"a": "same text here", "b": "same text here"}})]

    def brain(request):
        n = len([m for m in request.messages if m.role == "tool"])
        if request.node_id == "n3":
            from aichain.gateways.llm.scripted import tool_response
            return tool_response(calls[n] if n < len(calls) else ("done", {"summary": "ok"}))
        from aichain.sim.brains import persona_brain
        return persona_brain(request)

    net = build_network("scripted", brain=brain, workshop=Workshop(dev_exec, tmp_path / "w.db"))
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    net.advance(SHORT_PARAMS.commit_blocks + SHORT_PARAMS.reveal_blocks + SHORT_PARAMS.judge_blocks)
    results = [e for e in net.events if e["event"] == "tool" and e["node"] == "n3"]
    assert '"answerer": "n2"' in results[0]["result"]
    assert '"value": 1024' in results[1]["result"] and "near-copy" in results[2]["result"]
    assert net.board.licenses == set()  # free tools: nothing bought
    prompt = net.gateway.adapters["scripted"].requests[-1].messages[1].content
    assert "lib.collusion-scan" in prompt and "무료 표준 도구" in prompt


def test_history_tool_without_workshop():
    net = build_network("scripted")
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    out = json.loads(net.nodes["n3"]._exec_tool("history", json.dumps({"what": "bounties"})).split("\n", 1)[1].rsplit("\n", 1)[0])
    assert {r["answerer"] for r in out} == {"n2", "n4"} and out[0]["title"]
