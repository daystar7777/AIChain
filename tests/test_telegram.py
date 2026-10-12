import json

import httpx
from fastapi.testclient import TestClient

from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import TokenStore, add_client
from aichain.gateways.llm.scripted import tool_response
from aichain.manager import ApprovalBook
from aichain.sim.brains import persona_brain
from aichain.sim.factory import build_network
from aichain.telegram_bridge import TelegramBridge

TOKEN = "123456:" + "A" * 35


def make(tmp_path, brain=persona_brain):
    net = build_network("scripted", brain=brain)
    net.approvals = ApprovalBook()
    path = tmp_path / "t.json"
    add_client(path, "c", 5)
    admin = TestClient(create_app(NetworkRuntime(net), TokenStore.load(path), start_runtime=False,
                                  admin_token="adm"), headers={"Authorization": "Bearer adm"})
    sent, updates = [], []

    def handler(request):
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        if method == "getUpdates":
            out, updates[:] = list(updates), []
            return httpx.Response(200, json={"ok": True, "result": out})
        sent.append((body.get("chat_id"), body.get("text")))
        return httpx.Response(200, json={"ok": True, "result": {}})

    tg = TelegramBridge(TOKEN, "42", admin, tg=httpx.Client(transport=httpx.MockTransport(handler)),
                        state_path=tmp_path / "s.json")
    return net, tg, sent, updates


def msg(uid, chat, text):
    return {"update_id": uid, "message": {"chat": {"id": chat}, "text": text}}


def test_only_the_manager_chat_is_served(tmp_path):
    net, tg, sent, updates = make(tmp_path)
    updates += [msg(1, 99, "/status"), msg(2, 42, "/status")]
    tg.poll_once()
    assert sent[0][0] == "99" and "chat id: 99" in sent[0][1] and "블록" not in sent[0][1]
    assert sent[1][0] == "42" and sent[1][1].startswith("블록 #")


def test_approval_flow_over_telegram(tmp_path):
    net, tg, sent, updates = make(tmp_path)
    net.nodes["n1"]._exec_tool("transfer", json.dumps({"to": "n2", "amount_P": 4, "memo": "협업 보상"}))
    tg.push_notifications()
    assert "새 승인 요청" in sent[-1][1] and "협업 보상" in sent[-1][1]
    aid = net.approvals.pending()[0].approval_id
    updates.append(msg(3, 42, f"/approve {aid} 좋아요"))
    tg.poll_once()
    assert "EXECUTED" in sent[-1][1] and net.board.accounts["n2"].balance_mP == 1_004_000
    tg.push_notifications()  # already seen: no duplicate
    assert sum("새 승인 요청" in t for _, t in sent) == 1


def test_manager_chats_with_a_node(tmp_path):
    def brain(request):
        user = next(m.content for m in reversed(request.messages) if m.role == "user")
        if "manager_msg" in user and not any(m.role == "tool" for m in request.messages):
            assert '<data source="manager">' in user
            return tool_response(("notify_manager", {"text": "오늘은 MCP 보안을 공부했습니다."}),
                                 ("done", {"summary": "답함"}))
        return persona_brain(request)

    net, tg, sent, updates = make(tmp_path, brain)
    updates.append(msg(4, 42, "/ask n1 오늘 뭐 배웠어?"))
    tg.poll_once()
    assert "전달했습니다" in sent[-1][1]
    net.run_until_idle()
    tg.push_notifications()
    assert sent[-1][1] == "💬 n1: 오늘은 MCP 보안을 공부했습니다."


def test_bad_command_does_not_crash(tmp_path):
    net, tg, sent, updates = make(tmp_path)
    updates += [msg(5, 42, "/approve nope"), msg(6, 42, "/whatever")]
    tg.poll_once()
    assert "실패 409" in sent[0][1] and "모르는 명령" in sent[1][1]


def test_bot_token_is_redacted_in_the_journal():
    from aichain.journal.store import redact
    assert TOKEN not in json.dumps(redact({"x": f"token {TOKEN}", "env": f"TELEGRAM_BOT_TOKEN={TOKEN}"}))
