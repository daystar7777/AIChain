import json

from aichain.gateways.web import FakeWebProvider, SearchGateway
from aichain.mastermind.tools import ALWAYS, EVENT_TOOLS, tool_specs
from aichain.node import QueuedEvent
from aichain.sim.factory import build_network
from aichain.workshop.registry import Workshop


def test_tools_are_scoped_by_event():
    net = build_network("scripted", web=SearchGateway(FakeWebProvider()), workshop=Workshop())
    n2 = net.nodes["n2"]
    all_tools = n2.tool_names()
    n2.current_event = QueuedEvent(3, 0, "bounty_new", {})
    scoped = n2.tool_names()
    n2.current_event = None
    assert "answer" in scoped and "loans" not in scoped and "propose" not in scoped
    assert set(scoped) <= set(all_tools) and ALWAYS & set(all_tools) <= set(scoped)
    size = lambda names: len(json.dumps([t.parameters for t in tool_specs(names)]))
    assert size(scoped) < size(all_tools) / 2
    # A tool outside the scope is refused during that event.
    n2.current_event = QueuedEvent(3, 0, "bounty_new", {})
    assert "지금 쓸 수 없는 도구" in n2._exec_tool("loans", "{}")
    n2.current_event = None


def test_economy_ticks_are_staggered_and_summarise_the_node():
    net = build_network("scripted")
    net.economy_interval_blocks = 80
    net.advance(80)
    ticks = [(e["height"], e["node"]) for e in net.events
             if e["event"] == "episode_start" and e["kind"] == "economy_tick"]
    assert sorted(n for _, n in ticks) == ["n1", "n2", "n3", "n4"] and len({h for h, _ in ticks}) == 4
    summary = net.nodes["n1"]._economy_summary()
    assert "오늘 남은 할당량" in summary and "대출 요청" in summary and "저장 공간" in summary
    n2 = net.nodes["n2"]
    from aichain.common import sha256_hex
    o = net.board.create_offer("n1", "info", "AI 동향", "h", 1000, sha256_hex("x"), 720)
    assert o.offer_id in n2._economy_summary()
    assert o.offer_id in n2._exec_tool("offers", json.dumps({"query": "{}"}))
    assert "economy_tick" in EVENT_TOOLS and "savings" in EVENT_TOOLS["economy_tick"]


def test_ratings_for_every_kind_of_deal():
    import pytest
    from aichain.board import Board, BoardError, Params
    from aichain.common import sha256_hex

    b = Board(Params(issuance=False))
    for n in ("a", "b", "c"):
        b.open_account(n, 100_000)
    o = b.create_offer("a", "info", "t", "h", 1000, sha256_hex("x"), 720)
    pu = b.buy_offer("b", o.offer_id)
    with pytest.raises(BoardError):
        b.rate("b", "a", 1, pu.purchase_id)  # not settled yet
    b.deliver_offer("a", pu.purchase_id, "c")
    b.confirm_delivery("b", pu.purchase_id, True, "r")
    b.rate("b", "a", 2, pu.purchase_id)
    loan = b.create_loan_request("c", 5000, 0, 0, 720, "t")
    b.fund_loan("a", loan.loan_id)
    b.repay_loan("c", loan.loan_id)
    b.rate("a", "c", 1, loan.loan_id)
    with pytest.raises(BoardError):
        b.rate("b", "c", 1, loan.loan_id)  # b was not part of the loan
    assert len(b.ratings) == 2


def test_tool_disputes_refund_and_withdraw():
    from aichain.board import Board, Params
    from aichain.reputation import for_board

    b = Board(Params(issuance=False))
    for n in ("author", "buyer"):
        b.open_account(n, 100_000)
    b.publish_tool("author", "author.x", 1, "h", 20_000, 0)
    b.license_tool("buyer", "author.x", 1)
    d = b.open_dispute("buyer", "tool", "author.x@1", "output differs from the description")
    assert d.bond_mP == 2000
    b.resolve_dispute("root", d.dispute_id, "uphold", "")
    assert b.accounts["buyer"].balance_mP == 100_000 and b.tools[("author.x", 1)].withdrawn
    assert for_board(b, "author").terms_e6["T"] < 500_000 and b.total_supply_check()


def test_self_audit_finds_broken_books_and_alerts():
    from aichain import audit
    from aichain.board import Board, Params

    b = Board(Params(issuance=False))
    b.open_account("a", 10_000)
    assert audit.run(b) == []
    b.accounts["a"].balance_mP += 5  # minted from nowhere
    b.accounts["a"].locked_mP = 3
    problems = audit.run(b)
    assert any("supply mismatch" in p for p in problems) and any("sum of locks" in p for p in problems)


def test_audit_runs_hourly_on_the_network():
    net = build_network("scripted")
    net.audit_interval_blocks = 50
    net.advance(50)
    assert [e["ok"] for e in net.events if e["event"] == "audit"] == [True]
    net.board.accounts["n1"].balance_mP += 1
    net.advance(50)
    assert any(e["event"] == "audit_alert" for e in net.events)
    assert "자기 감사" in net.manager_outbox[-1]["text"]


def test_bridge_records_and_policy_history_survive_restart(tmp_path):
    from aichain.bridge.runtime import NetworkRuntime

    path = tmp_path / "bridge_state.json"
    net = build_network("scripted")
    rt = NetworkRuntime(net, state_path=path)
    rec = rt.submit("client", "t", "b", "c")
    net.nodes["n1"].policy.record_spend(1234, net.board.height)
    rt.save_state()
    net2 = build_network("scripted")
    rt2 = NetworkRuntime(net2, state_path=path)
    q = rt2.question(rec.ref)
    assert q["status"] == "failed" and "restarted" in q["note"]
    assert net2.nodes["n1"].policy.spent_last_hour(net2.board.height) == 1234


def test_journal_drops_old_prompts_but_keeps_responses(tmp_path):
    import time

    from aichain.journal.store import Journal, connect

    j = Journal(tmp_path / "j.db", keep_prompts_days=14)
    j.start_run("t")
    j.record("llm_call", {"messages": [{"role": "user", "content": "long prompt"}],
                          "response": {"text": "ok"}})
    j.record("tool", {"tool": "x"})
    assert j.prune(now=time.time() + 13 * 86400) == 0
    assert j.prune(now=time.time() + 15 * 86400) == 1
    rows = {k: json.loads(d) for k, d in connect(tmp_path / "j.db").execute("SELECT kind, data FROM entries")}
    assert "messages" not in rows["llm_call"] and rows["llm_call"]["messages_pruned"] is True
    assert rows["llm_call"]["response"] == {"text": "ok"} and rows["tool"] == {"tool": "x"}


def test_operator_can_queue_a_tick(tmp_path):
    from fastapi.testclient import TestClient

    from aichain.bridge.runtime import NetworkRuntime
    from aichain.bridge.server import create_app
    from aichain.bridge.tokens import TokenStore

    net = build_network("scripted")
    c = TestClient(create_app(NetworkRuntime(net), TokenStore.load(tmp_path / "t.json"), admin_token="adm"),
                   headers={"Authorization": "Bearer adm"})
    assert c.post("/admin/v1/nodes/n2/tick", json={"kind": "economy_tick"}).json() == {"queued": "economy_tick"}
    assert net.nodes["n2"].has_pending
    assert c.post("/admin/v1/nodes/zz/tick", json={"kind": "economy_tick"}).status_code == 404
    assert c.post("/admin/v1/nodes/n2/tick", json={"kind": "transfer"}).status_code == 422


def test_admin_action_while_network_is_busy_is_accepted(tmp_path):
    from fastapi.testclient import TestClient

    from aichain.bridge.runtime import NetworkRuntime
    from aichain.bridge.server import create_app
    from aichain.bridge.tokens import TokenStore

    rt = NetworkRuntime(build_network("scripted"))

    def busy(fn, timeout=30.0):
        raise TimeoutError()
    rt.call = busy
    c = TestClient(create_app(rt, TokenStore.load(tmp_path / "t.json"), admin_token="adm"),
                   headers={"Authorization": "Bearer adm"})
    r = c.post("/admin/v1/nodes/n2/tick", json={"kind": "economy_tick"})
    assert r.status_code == 202 and r.json()["accepted"] is True


def test_nodes_can_ask_peers_from_research_twice_a_day():
    net = build_network("scripted", web=SearchGateway(FakeWebProvider()))
    n1 = net.nodes["n1"]
    n1.current_event = QueuedEvent(5, 0, "research_tick", {})
    assert "ask" in n1.tool_names()
    q = {"question": {"title": "t", "body": "b", "acceptance_criteria": "c"}, "reward_P": 5, "tags": ["x"]}
    for _ in range(2):
        assert "현상금 생성" in n1._exec_tool("ask", json.dumps(q))
        n1.policy.start_episode()
    assert "2/day" in n1._exec_tool("ask", json.dumps(q))
    # A Bridge client's question is not the node's own and is not limited this way.
    n1.policy.start_episode()
    n1.current_event = QueuedEvent(1, 0, "bridge_question", {})
    assert "현상금 생성" in n1._exec_tool("ask", json.dumps(q))
    n1.current_event = None
    assert n1.policy.dump()["self_asks"]
    assert "다른 노드의 관심사" in n1.render_event(QueuedEvent(5, 0, "research_tick", {}))


def test_manager_is_told_when_llm_calls_keep_failing():
    from aichain.provider_watch import ProviderWatch

    net = build_network("scripted")
    now = [0.0]
    ProviderWatch(net, clock=lambda: now[0])
    end = lambda **kw: net.log("n1", "episode_end", kind="research_tick", **kw)
    end(stopped="provider_error", llm_calls=0, summary="timeout")
    assert not net.manager_outbox  # one transient failure is not worth a message
    end(stopped="provider_error", llm_calls=0, summary='HTTP 401: {"code":"api_key_expired"}')
    assert len(net.manager_outbox) == 1 and "API 키" in net.manager_outbox[0]["text"]
    end(stopped="provider_error", llm_calls=0, summary="HTTP 401")
    assert len(net.manager_outbox) == 1  # not repeated within 6 hours
    now[0] = 7 * 3600
    end(stopped="provider_error", llm_calls=0, summary="HTTP 401")
    assert len(net.manager_outbox) == 2
    end(stopped="done", llm_calls=2, summary="ok")
    assert "다시 됩니다" in net.manager_outbox[-1]["text"]
    end(stopped="done", llm_calls=2, summary="ok")
    assert len(net.manager_outbox) == 3
