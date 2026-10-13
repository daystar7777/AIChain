"""B1: a share of the daily LLM-call caps is kept for duties, and refused duties are retried."""

import re
import time

import pytest

from aichain import setup as node_setup
from aichain.board import BountyState
from aichain.bridge.runtime import NetworkRuntime
from aichain.gateways.llm.base import (
    CallLimitReached,
    GatewayConfig,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderError,
    TokenBucket,
    Usage,
)
from aichain.gateways.llm.scripted import ScriptedAdapter, tool_response
from aichain.sim.brains import persona_brain
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS as P, build_network, default_nodes
from aichain.sim.network import Network

TO_JUDGING = P.commit_blocks + P.reveal_blocks + 1


def today():
    return time.strftime("%Y-%m-%d", time.gmtime())


class One:
    name = "x"

    def complete(self, request):
        return LLMResponse("ok", [], Usage(1, 1))


def gateway(**cfg):
    gw = LLMGateway({"x": One()}, GatewayConfig(**cfg))
    for n in ("a", "b"):
        gw.register(n, TokenBucket(10**6, lambda: 0))
    return gw


def call(gw, node, priority):
    gw.chat("x", LLMRequest(node, "m", [Message("user", "hi")], []), priority)


def test_optional_work_stops_at_the_reserve_line_and_duties_use_the_rest():
    gw = gateway(daily_calls_per_node=10, daily_calls_total=100)  # default 20%: 2 calls kept
    for _ in range(8):
        call(gw, "a", 5)  # research, economy tick, reflection
    with pytest.raises(CallLimitReached, match="optional work"):
        call(gw, "a", 5)
    assert gw.remaining("a", "llm_calls", duty=False) == 0 and gw.remaining("a", "llm_calls") == 2
    assert not gw.has_daily_room("a", 5) and gw.has_daily_room("a", 1) and gw.has_daily_room("a")
    call(gw, "a", 1)  # judging
    call(gw, "a", 3)  # a Bridge question
    with pytest.raises(CallLimitReached, match="daily call cap 10 reached"):
        call(gw, "a", 1)
    call(gw, "b", 5)  # other nodes are not affected


def test_the_total_cap_keeps_a_reserve_too():
    gw = gateway(daily_calls_total=5, duty_reserve_share=0.4)  # 2 of 5 for duties
    call(gw, "a", 5)
    call(gw, "b", 4)
    call(gw, "a", 5)
    with pytest.raises(CallLimitReached, match="on the network"):
        call(gw, "b", 5)
    call(gw, "b", 2)  # a vote
    call(gw, "a", 1)  # judging
    with pytest.raises(CallLimitReached, match="all nodes"):
        call(gw, "b", 1)


def test_zero_share_disables_the_reserve_and_the_reserve_is_not_for_rent():
    gw = gateway(daily_calls_per_node=10, duty_reserve_share=0)
    for _ in range(10):
        call(gw, "a", 5)
    gw = gateway(daily_calls_per_node=10)
    for _ in range(5):
        call(gw, "a", 5)
    with pytest.raises(CallLimitReached, match="duty reserve"):
        gw.transfer_quota("llm_calls", "a", "b", 4)  # 5 left, 2 of them kept for duties
    gw.transfer_quota("llm_calls", "a", "b", 3)
    assert gw.remaining("a", "llm_calls", duty=False) == 0 and gw.remaining("a", "llm_calls") == 2
    assert gw.remaining("b", "llm_calls", duty=False) == 11  # rented calls are all for optional work


def test_duties_still_run_after_optional_work_used_up_the_day():
    net = build_network("scripted", interests=node_setup.DEFAULT_INTERESTS,
                        daily_calls_per_node=50, daily_calls_total=200)
    net.economy_interval_blocks = 4
    gw = net.gateway
    for nid in net.nodes:  # research and ticks used everything above the reserve
        gw._daily[(today(), nid)] = 50 - gw.reserve(nid)
    net.research()
    net.reflect()
    net.advance(4)
    assert not any(e.event in ("research_tick", "reflect", "economy_tick") for e in net.episodes)
    assert "LLM 호출 0 (그 밖에 판정·투표·질문 같은 의무에만 쓰는 예비 10)" in \
        net.nodes["n1"]._system_messages("x")[1].content
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    net.advance(TO_JUDGING + 2)
    (b,) = net.board.bounties.values()
    assert b.state == BountyState.PAID and net.board.answers[b.accepted].author == "n2"
    assert all(e.stopped == "done" for e in net.episodes)


def test_refused_judging_waits_for_capacity_and_runs_later():
    net = build_network("scripted", daily_calls_total=4)
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()  # n1 asks, n2 and n4 answer: three calls
    net.gateway._daily[(today(), "*")] = 4  # the rest of today's calls went elsewhere
    net.advance(TO_JUDGING)
    (b,) = net.board.bounties.values()
    assert b.state == BountyState.JUDGING
    assert [(e.event, e.stopped) for e in net.episodes][-1] == ("judge_due", "call_limit")
    n1 = net.nodes["n1"]
    assert [w.kind for w in n1.retry.waiting] == ["judge_due"]
    episodes = len(net.episodes)
    net.advance(5)
    assert len(net.episodes) == episodes and len(n1.retry.waiting) == 1  # waiting costs nothing
    net.gateway._daily.clear()  # a new UTC day
    net.advance(1)
    assert b.state == BountyState.PAID and net.board.answers[b.accepted].author == "n2"
    assert not n1.retry.waiting
    assert any(e["event"] == "duty_deferred" and e["kind"] == "judge_due" for e in net.events)


def test_a_waiting_duty_that_is_no_longer_due_is_dropped_and_logged():
    net = build_network("scripted", daily_calls_total=4)
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    net.gateway._daily[(today(), "*")] = 4
    net.advance(TO_JUDGING + P.judge_blocks + 1)  # the judging window closes: AUTO_SPLIT
    (b,) = net.board.bounties.values()
    assert b.outcome == BountyState.AUTO_SPLIT and not net.nodes["n1"].retry.waiting
    assert any(e["event"] == "duty_dropped" and e["reason"] == "no longer due" for e in net.events)


def test_a_vote_is_queued_and_retried_even_when_the_voter_had_no_calls_left():
    def brain(request):
        user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if request.messages[-1].role == "tool":
            return tool_response(("done", {"summary": "ok"}))
        if "idle_tick" in user:
            return tool_response(("propose", {"kind": "param", "title": "판정 기간", "spec": "길게",
                                              "changes": [{"key": "judge_blocks", "to": 96}]}))
        if "proposal_open" in user:
            pid = re.search(r'proposal_id="([^"]+)"', user).group(1)
            return tool_response(("vote", {"proposal_id": pid, "choice": "yes", "reason": "자동 분배가 줄어 판정 품질이 오른다"}))
        return persona_brain(request)

    net = build_network("scripted", brain=brain, daily_calls_per_node=10)
    net.gateway._daily[(today(), "n2")] = 10  # n2 spent its whole day, reserve included
    net.nodes["n1"].enqueue("idle_tick", {})
    net.run_until_idle()
    (prop,) = net.board.proposals.values()
    assert set(prop.votes) == {"n3", "n4"} and [w.kind for w in net.nodes["n2"].retry.waiting] == ["proposal_open"]
    net.gateway._daily[(today(), "n2")] = 0
    net.advance(1)
    assert prop.votes["n2"] == "yes"


def test_provider_outage_is_retried_with_backoff_but_adapter_bugs_are_not():
    failing = {"n1": 2}

    def brain(request):
        if request.node_id == "n1" and failing["n1"] > 0:
            failing["n1"] -= 1
            raise ProviderError("upstream 503", status=503)
        return persona_brain(request)

    adapter = ScriptedAdapter(brain)
    net = Network(LLMGateway({"scripted": adapter}, GatewayConfig(max_retries=0)), P)
    for cfg in default_nodes("scripted", "scripted", None):
        net.add_node(cfg)
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    assert net.episodes[-1].stopped == "provider_error" and not net.board.bounties
    net.advance(30)
    assert not net.board.bounties  # backing off
    net.advance(60)
    assert not net.board.bounties  # second outage, backing off again
    net.advance(61)
    (b,) = net.board.bounties.values()  # the third attempt asks
    assert b.asker == "n1"


def test_bridge_question_waiting_for_capacity_is_reported_as_queued_not_failed():
    net = build_network("scripted", daily_calls_per_node=10)
    runtime = NetworkRuntime(net)
    net.gateway._daily[(today(), "n1")] = 10
    rec = runtime.submit("c", DEFAULT_QUESTION["title"], DEFAULT_QUESTION["body"],
                         DEFAULT_QUESTION["acceptance_criteria"])
    runtime._drain_inbox()
    net.run_until_idle()
    q = runtime.question(rec.ref)
    assert q["status"] == "queued" and "waiting for LLM capacity" in q["note"]
    net.gateway._daily.clear()
    net.advance(1)
    assert [e["event"] for e in rec.events][:5] == ["queued", "processing", "queued", "processing", "bounty_open"]
