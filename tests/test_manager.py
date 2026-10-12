import json

import pytest
from fastapi.testclient import TestClient

from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.manager import ApprovalBook, ManagerPolicy
from aichain.sim.factory import build_network


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def setup(tmp_path=None):
    clock = Clock()
    net = build_network("scripted")
    net.approvals = ApprovalBook(ManagerPolicy(), tmp_path / "a.json" if tmp_path else None, clock=clock)
    return net, clock


def test_policy_rules():
    with pytest.raises(ValueError):
        ManagerPolicy(timeout_hours=24)
    p = ManagerPolicy()
    assert p.needs_approval("transfer", 1, 1_000_000)
    assert not p.needs_approval("offer_buy", 5_000, 1_000_000)
    assert p.needs_approval("offer_buy", 150_000, 1_000_000)  # > 100P
    assert p.needs_approval("ask", 30_000, 100_000)  # > 20% of balance


def test_transfer_waits_for_approval_then_executes(tmp_path):
    net, clock = setup(tmp_path)
    out = net.nodes["n1"]._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 10, "memo": "도움 감사"}))
    assert "관리인 승인 대기" in out and net.board.accounts["n2"].balance_mP == 1_000_000
    (a,) = net.approvals.pending()
    net.approvals.decide(a.approval_id, True, "operator")
    net.run_approvals()
    assert a.state == "EXECUTED" and net.board.accounts["n2"].balance_mP == 1_010_000
    assert any(m.source == f"manager:{a.approval_id}" for m in net.nodes["n1"].memory.records)
    # persisted
    again = ApprovalBook(ManagerPolicy(), tmp_path / "a.json")
    assert again.items[a.approval_id].state == "EXECUTED"


def test_denied_is_not_executed_and_timeout_executes():
    net, clock = setup()
    n1 = net.nodes["n1"]
    n1._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 5, "memo": "a"}))
    n1._exec_tool("transfer", json.dumps({"to": "n3", "amount_P": 7, "memo": "b"}))
    first, second = net.approvals.pending()
    net.approvals.decide(first.approval_id, False, "operator", "no")
    net.run_approvals()
    assert first.state == "DENIED" and net.board.accounts["n2"].balance_mP == 1_000_000
    clock.t += 71 * 3600
    net.run_approvals()
    assert second.state == "PENDING"
    clock.t += 2 * 3600  # past 72h without an answer
    net.run_approvals()
    assert second.state == "EXECUTED" and second.decided_by == "timeout"
    assert net.board.accounts["n3"].balance_mP == 1_007_000


def test_small_spends_are_not_held_and_limits_are_rechecked():
    net, clock = setup()
    n1 = net.nodes["n1"]
    out = n1._exec_tool("loan_request", json.dumps({"principal_P": 5, "interest_pct": 0, "term_hours": 1,
                                                     "purpose": "x"}))
    assert "관리인" not in out
    # A transfer approved later is still checked against Policy Guard at execution time.
    n1._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 500, "memo": "big"}))
    (a,) = net.approvals.pending()
    net.approvals.decide(a.approval_id, True, "operator")
    net.run_approvals()
    assert a.state == "FAILED" and "Policy Guard" in a.result


def test_operator_endpoints(tmp_path):
    net, clock = setup()
    net.nodes["n1"]._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 3, "memo": "m"}))
    path = tmp_path / "t.json"
    add_client(path, "c", 5)
    client = TestClient(create_app(NetworkRuntime(net), TokenStore.load(path), start_runtime=False,
                                   admin_token="adm"))
    hdr = {"Authorization": "Bearer adm"}
    (item,) = client.get("/admin/v1/approvals", headers=hdr).json()
    r = client.post(f"/admin/v1/approvals/{item['approval_id']}", headers=hdr, json={"decision": "approve"})
    assert r.status_code == 200 and net.approvals.items[item["approval_id"]].state == "EXECUTED"
    assert client.post(f"/admin/v1/approvals/{item['approval_id']}", headers=hdr,
                       json={"decision": "deny"}).status_code == 409
