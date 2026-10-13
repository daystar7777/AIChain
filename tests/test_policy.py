import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aichain import admin
from aichain.board import BountyState
from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.manager import ApprovalBook, ManagerPolicy
from aichain.mastermind.policy import BLOCKS_PER_HOUR, PolicyGuard, PolicyViolation
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import build_network

OVERSIZED_ASK = json.dumps({"question": DEFAULT_QUESTION, "reward_P": 150, "tags": []})  # > 10% cap


def operator_client(tmp_path, net, state_path=None):
    path = tmp_path / "t.json"
    add_client(path, "c", 5)
    runtime = NetworkRuntime(net, state_path=state_path)
    app = create_app(runtime, TokenStore.load(path), start_runtime=False, admin_token="adm")
    return TestClient(app, headers={"Authorization": "Bearer adm"}), runtime


def n1_requests(net):
    return [r for r in net.gateway.adapters["scripted"].requests if r.node_id == "n1"]


def test_outflow_rule_uses_the_balance_at_the_window_start():
    g = PolicyGuard("x")
    alerts = []
    g.on_pause = alerts.append
    g.observe_balance(0, 1_000_000)
    g.observe_balance(100, 2_000_000)  # income during the hour does not raise the base
    g.check_spend(100_000, 2_000_000, 100, bounty=False)
    g.record_spend(100_000, 100)
    with pytest.raises(PolicyViolation, match="50% of the balance at the window start"):
        g.check_spend(450_000, 1_900_000, 200, bounty=False)  # 550P moved > 500P (50% of 1000P)
    assert g.paused and len(alerts) == 1 and "1000000mP" in alerts[0]
    # A later hour starts from the newer balance; the pause itself waits for the operator.
    assert g.balance_at_window_start(100 + BLOCKS_PER_HOUR) == 2_000_000
    with pytest.raises(PolicyViolation, match="paused"):
        g.check_write()
    # Dust never triggers it (a node that ran low and earned a little).
    h = PolicyGuard("y")
    h.observe_balance(0, 4_000)
    h.check_spend(3_000, 20_000, 10, bounty=False)
    assert not h.paused


def test_pause_state_survives_a_restart():
    g = PolicyGuard("x")
    g.observe_balance(5, 1_000_000)
    g.pause("test reason")
    again = PolicyGuard("x")
    again.restore(json.loads(json.dumps(g.dump())))
    assert again.paused and again.pause_reason == "test reason" and again.balance_marks == [(5, 1_000_000)]
    again.unpause()
    assert not again.paused and again.pause_reason is None and again.consecutive_violations == 0


def test_repeated_violations_pause_alert_and_stop_llm_calls(tmp_path):
    net = build_network("scripted")
    n1 = net.nodes["n1"]
    for _ in range(10):
        assert "Policy Guard" in n1._exec_tool("ask", OVERSIZED_ASK)
    assert n1.policy.paused and "10 consecutive violations" in n1.policy.pause_reason
    assert [e["node"] for e in net.events if e["event"] == "policy_paused"] == ["n1"]
    alert = net.manager_outbox[-1]
    assert alert["node"] == "core" and "n1" in alert["text"] and "unpause n1" in alert["text"]
    # No LLM calls while paused, for any event; the Bridge client hears that its question failed.
    client, runtime = operator_client(tmp_path, net)
    rec = runtime.submit("c", **DEFAULT_QUESTION)
    runtime._drain_inbox()
    n1.enqueue("idle_tick", {})
    net.run_until_idle()
    assert n1_requests(net) == [] and runtime.question(rec.ref)["status"] == "failed"
    assert {e["kind"] for e in net.events if e["event"] == "skip_event"} == {"bridge_question", "idle_tick"}
    assert client.get("/admin/v1/status").json()["nodes"][0]["paused"] is True


def test_judging_waits_for_the_operator_then_resumes(tmp_path, monkeypatch):
    state = tmp_path / "bridge_state.json"
    net = build_network("scripted")
    net.bridge_question("n1", "test", **DEFAULT_QUESTION)
    net.run_until_idle()
    n1 = net.nodes["n1"]
    n1.policy.pause("operator test")
    net.advance(24 + 12 + 1)  # commit + reveal: judge_due is raised and skipped
    (b,) = net.board.bounties.values()
    assert b.state == BountyState.JUDGING and len(n1_requests(net)) == 1  # only the original ask
    assert any(e["event"] == "skip_event" and e["kind"] == "judge_due" for e in net.events)

    client, runtime = operator_client(tmp_path, net, state_path=state)
    monkeypatch.setattr(admin.httpx, "Client", lambda base_url, headers, timeout: client)
    token = tmp_path / "admin_token"
    token.write_text("adm")
    assert admin.main(["--token-file", str(token), "unpause", "n1", "--note", "checked"]) == 0
    assert not n1.policy.paused and json.loads(state.read_text())["policy"]["n1"]["paused"] is False
    assert any(e["event"] == "policy_unpaused" and e["note"] == "checked" for e in net.events)
    assert client.post("/admin/v1/nodes/n1/unpause", json={}).status_code == 409  # not paused
    assert client.post("/admin/v1/nodes/n9/unpause", json={}).status_code == 404
    net.run_until_idle()  # judge_due was raised again on unpause
    assert b.accepted is not None and len(n1_requests(net)) == 2


def test_a_pause_mid_episode_ends_it():
    from aichain.gateways.llm.scripted import tool_response

    def stubborn(request):
        return tool_response(*[("ask", {"question": DEFAULT_QUESTION, "reward_P": 150, "tags": []})] * 2)

    net = build_network("scripted", brain=stubborn)
    n1 = net.nodes["n1"]
    n1.policy.consecutive_violations = 8
    n1.enqueue("idle_tick", {})
    net.run_until_idle()
    (ep,) = net.episodes
    assert ep.stopped == "paused" and ep.llm_calls == 1 and n1.policy.paused


def test_attempt_to_move_half_the_balance_pauses_even_when_held_for_the_manager():
    net = build_network("scripted")
    net.approvals = ApprovalBook(ManagerPolicy())
    n1 = net.nodes["n1"]
    out = n1._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 600, "memo": "전부"}))
    assert "Policy Guard" in out and n1.policy.paused and not net.approvals.pending()
    assert "50%" in net.manager_outbox[-1]["text"] and net.board.accounts["n2"].balance_mP == 1_000_000
    # Splitting it into several held transfers does not get around the rule.
    net = build_network("scripted")
    net.approvals = ApprovalBook(ManagerPolicy())
    n1 = net.nodes["n1"]
    assert "관리인 승인 대기" in n1._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 300, "memo": "a"}))
    assert "Policy Guard" in n1._exec_tool("transfer", json.dumps({"to": "n3", "amount_P": 250, "memo": "b"}))
    assert n1.policy.paused and len(net.approvals.pending()) == 1


def test_paused_node_does_not_execute_unanswered_approvals():
    class Clock:
        t = 1000.0

        def __call__(self):
            return self.t

    clock = Clock()
    net = build_network("scripted")
    net.approvals = ApprovalBook(ManagerPolicy(), clock=clock)
    n1 = net.nodes["n1"]
    n1._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 5, "memo": "a"}))
    n1._exec_tool("transfer", json.dumps({"to": "n3", "amount_P": 7, "memo": "b"}))
    first, second = net.approvals.pending()
    n1.policy.pause("review")
    net.approvals.decide(first.approval_id, True, "operator")  # the human said yes: executes
    clock.t += 73 * 3600  # the other one times out while the node is paused: does not
    net.run_approvals()
    assert first.state == "EXECUTED" and net.board.accounts["n2"].balance_mP == 1_005_000
    assert second.state == "FAILED" and "일시 정지" in second.result
    assert net.board.accounts["n3"].balance_mP == 1_000_000


def test_unpause_over_telegram(tmp_path):
    from aichain.telegram_bridge import TelegramBridge

    net = build_network("scripted")
    net.nodes["n2"].policy.pause("too many violations")
    client, _ = operator_client(tmp_path, net)
    tg = TelegramBridge("123456:" + "A" * 35, "42", client,
                        tg=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))))
    assert "⛔ 일시 정지 (/unpause n2)" in tg.handle("/status")
    assert "해제" in tg.handle("/unpause n2 확인함") and not net.nodes["n2"].policy.paused
    assert "실패 409" in tg.handle("/unpause n2")
