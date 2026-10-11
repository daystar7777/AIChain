"""Fault injection: each test breaks one thing on purpose and checks the network survives it
(or, for the chain log, stops safely)."""

import json
import sqlite3
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.chain.store import ChainStore
from aichain.gateways.llm.base import LLMRequest, Message, ProviderError
from aichain.gateways.llm.opencode_go import OpenCodeGoAdapter, parse_output
from aichain.gateways.llm.scripted import tool_response
from aichain.gateways.web import SearchGateway, WebError
from aichain.sim.brains import persona_brain
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS, build_network


def ask(net):
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()


# ---- episodes and tools ----

def test_bug_inside_a_tool_is_reported_and_the_episode_continues():
    net = build_network("scripted")
    real = net.nodes["n1"]._dispatch

    def broken(name, args):
        if name == "ask":
            raise KeyError("boom")
        return real(name, args)

    net.nodes["n1"]._dispatch = broken
    ask(net)
    (ep,) = [e for e in net.episodes if e.node_id == "n1"]
    assert ep.stopped == "done"  # the brain still called done after the error
    err = [e for e in net.events if e["event"] == "tool_error"]
    assert err and err[0]["tool"] == "ask" and "KeyError" in err[0]["error"]


def test_bug_in_episode_code_ends_only_that_episode():
    net = build_network("scripted")
    net.nodes["n1"].render_event = lambda ev: (_ for _ in ()).throw(RuntimeError("render bug"))
    ask(net)
    (ep,) = net.episodes
    assert ep.stopped == "error" and "render bug" in ep.summary
    assert net.current_episode is None and net.nodes["n1"].current_event is None
    assert any(e["event"] == "episode_error" for e in net.events)
    # The node still works for the next event.
    del net.nodes["n1"].render_event
    ask(net)
    assert net.episodes[-1].stopped == "done"


def test_adapter_bug_is_not_retried_and_does_not_stop_others():
    calls = []

    def flaky(request):
        calls.append(request.node_id)
        if request.node_id == "n1":
            raise ValueError("adapter bug")
        return persona_brain(request)

    net = build_network("scripted", brain=flaky)
    ask(net)
    assert calls.count("n1") == 1  # no retries for status 0
    assert net.episodes[0].stopped == "provider_error" and "adapter error" in net.episodes[0].summary
    net.nodes["n2"].enqueue("idle_tick", {})
    net.run_until_idle()
    assert net.episodes[-1].node_id == "n2" and net.episodes[-1].stopped == "done"


def test_three_consecutive_tool_failures_end_the_episode():
    def stubborn(request):
        return tool_response(("judge", {"bounty_id": "b_nope", "accept": "a_nope"}))

    net = build_network("scripted", brain=stubborn)
    ask(net)
    (ep,) = net.episodes
    assert ep.stopped == "tool_errors" and ep.steps == 3


def test_journal_failure_does_not_stop_the_network(capsys):
    net = build_network("scripted")

    def broken_listener(entry):
        raise sqlite3.OperationalError("database or disk is full")

    net.listeners.append(broken_listener)
    ask(net)
    net.advance(SHORT_PARAMS.commit_blocks + SHORT_PARAMS.reveal_blocks + SHORT_PARAMS.judge_blocks + 2)
    (b,) = net.board.bounties.values()
    assert b.outcome is not None and net.listener_errors > 0
    assert "disk is full" in capsys.readouterr().err


def test_event_storm_does_not_crash():
    net = build_network("scripted")
    for _ in range(3):
        net.nodes["n1"].enqueue("idle_tick", {})
    assert net.run_until_idle(max_rounds=0) is False
    assert any(e["event"] == "busy" for e in net.events)
    assert net.run_until_idle() is True


# ---- chain log ----

def test_chain_write_failure_halts_safely_and_restart_recovers(tmp_path):
    db = tmp_path / "chain.db"
    store = ChainStore(db)
    net = build_network("scripted", chain=store)
    before = net.board.state_root()
    real_append = store.append
    store.append = lambda kind, *a, **k: (
        (_ for _ in ()).throw(sqlite3.OperationalError("disk full")) if kind == "tx" else real_append(kind, *a, **k))
    ask(net)
    assert net.halted and "disk full" in net.halted
    assert net.episodes[0].stopped == "halted"
    height = net.board.height
    net.advance(10)
    assert net.board.height == height  # no blocks while halted
    # Restart from the log: the unrecorded bounty is gone, everything recorded is intact.
    again = build_network("scripted", chain=ChainStore(db))
    assert not again.halted and not again.board.bounties
    assert again.board.state_root() == before
    ask(again)
    assert len(again.board.bounties) == 1


def test_bridge_refuses_questions_while_halted(tmp_path):
    path = tmp_path / "t.json"
    token = add_client(path, "c", 5)
    net = build_network("scripted")
    net.board.halted = "chain log write failed: test"
    app = create_app(NetworkRuntime(net, block_seconds=0.01), TokenStore.load(path), start_runtime=False)
    client = TestClient(app)
    r = client.post("/bridge/v1/questions", json=DEFAULT_QUESTION,
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 503
    assert client.get("/bridge/v1/network", headers={"Authorization": f"Bearer {token}"}).json()["halted"]


def test_runtime_backs_off_and_recovers_from_errors():
    net = build_network("scripted")
    runtime = NetworkRuntime(net, block_seconds=0.01)
    real = net.run_until_idle
    state = {"n": 0}

    def sometimes_broken(*a, **k):
        state["n"] += 1
        if state["n"] <= 2:
            raise RuntimeError("transient")
        return real(*a, **k)

    net.run_until_idle = sometimes_broken
    runtime.start()
    deadline = time.time() + 10
    while time.time() < deadline and net.board.height < 3:
        time.sleep(0.05)
    runtime.stop()
    assert len(runtime.errors) == 2 and net.board.height >= 3


# ---- provider responses ----

@pytest.mark.parametrize("body,status", [
    ({"error": {"message": "overloaded"}}, 502),
    ({"status": "failed", "output": []}, 502),
    ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}, 422),
    ({"output": "nope"}, 502),
    ([1, 2], 502),
])
def test_bad_provider_bodies_raise_provider_errors(body, status):
    with pytest.raises(ProviderError) as e:
        parse_output(body)
    assert e.value.status == status


def test_lenient_parsing_of_odd_tool_calls():
    resp = parse_output({"output": [
        {"type": "function_call", "name": "done", "arguments": {"summary": "x"}},  # dict, no call_id
        "garbage",
        {"type": "function_call", "id": "fc_9", "name": "get_state"},
    ], "usage": {}})
    assert [c.call_id for c in resp.tool_calls] == ["call_0", "fc_9"]
    assert json.loads(resp.tool_calls[0].arguments) == {"summary": "x"}
    assert resp.tool_calls[1].arguments == "{}"


def test_rejected_replay_items_are_retried_without_them():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if any(i.get("type") == "reasoning" for i in body["input"]):
            return httpx.Response(400, text="reasoning item not found")
        return httpx.Response(200, json={"output": [{"type": "message", "content": [
            {"type": "output_text", "text": "ok"}]}], "usage": {"input_tokens": 1, "output_tokens": 1}})

    adapter = OpenCodeGoAdapter("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    raw = [{"type": "reasoning", "id": "rs_1"},
           {"type": "function_call", "call_id": "c1", "name": "get_state", "arguments": "{}"}]
    from aichain.gateways.llm.base import ToolCall
    msgs = [Message("user", "hi"),
            Message("assistant", None, [ToolCall("c1", "get_state", "{}")], raw=raw),
            Message("tool", "result", call_id="c1", name="get_state")]
    resp = adapter.complete(LLMRequest("n1", "m", msgs, []))
    assert resp.text == "ok" and len(seen) == 2
    assert [i.get("type") for i in seen[1]["input"]] == [None, "function_call", "function_call_output"]


# ---- web ----

def test_malformed_web_responses_become_web_errors():
    class Weird:
        def search(self, q, k):
            return {"results": [{"title": "ok", "url": "https://a.example/", "snippet": "s"},
                                {"title": "bad", "url": "javascript:alert(1)"}, "junk"]}

        def fetch(self, url, max_chars):
            return {"text": None}

    class Crashing:
        def search(self, q, k):
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")

        def fetch(self, url, max_chars):
            return ["not", "a", "dict"]

    gw = SearchGateway(Weird())
    assert [r["url"] for r in gw.search("n1", "q", 5)["results"]] == ["https://a.example/"]
    with pytest.raises(WebError):
        gw.fetch("n1", "https://a.example/", 100)
    gw2 = SearchGateway(Crashing())
    with pytest.raises(WebError):
        gw2.search("n1", "q", 5)
    with pytest.raises(WebError):
        gw2.fetch("n1", "https://a.example/", 100)


def test_research_is_not_queued_when_the_daily_cap_is_used_up():
    from aichain import setup as node_setup

    net = build_network("scripted", interests=node_setup.DEFAULT_INTERESTS, daily_calls_total=1)
    net.gateway._daily[(time.strftime("%Y-%m-%d", time.gmtime()), "*")] = 1
    net.research()
    assert not any(n.has_pending for n in net.nodes.values())
