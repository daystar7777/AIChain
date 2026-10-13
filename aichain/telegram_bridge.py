"""Telegram bridge for the node manager (노드 관리인).

    python -m aichain.telegram_bridge          # reads TELEGRAM_BOT_TOKEN, TELEGRAM_MANAGER_CHAT_ID

Long-polls the Telegram Bot API and talks to the running network through the operator API on
loopback. Only the configured chat is served; any other chat gets its chat id and nothing else.
Pushes: outflow approval requests, open disputes and proposals, node messages to the manager.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
HELP = """AIChain 관리인 명령
/status — 네트워크 요약
/approvals — 승인 대기 유출
/approve <id> [메모] · /deny <id> [메모]
/disputes — 열린 분쟁 · /resolve <id> uphold|reject|void [메모]
/proposals — 열린 제안 · /vote <id> yes|no|abstain <이유>
/ask <노드> <메시지> — 노드에게 말하기 (예: /ask n2 오늘 뭐 배웠어?)
/unpause <노드> [메모] — Policy Guard 일시 정지 해제
/report — 리포트 링크"""


class TelegramBridge:
    def __init__(self, token: str, chat_id: str | None, admin: httpx.Client,
                 tg: httpx.Client | None = None, state_path: Path | None = None,
                 viewer_url: str = "http://100.98.150.35:8790") -> None:
        self.token, self.chat_id = token, (str(chat_id) if chat_id else None)
        self.admin = admin
        self.tg = tg or httpx.Client(timeout=40)
        self.state_path = state_path
        self.viewer_url = viewer_url
        self.state: dict[str, Any] = {"offset": 0, "seen": [], "outbox_after": 0}
        if state_path and state_path.exists():
            self.state.update(json.loads(state_path.read_text()))

    # ---- plumbing ----

    def _save(self) -> None:
        if self.state_path:
            self.state["seen"] = self.state["seen"][-500:]
            self.state_path.write_text(json.dumps(self.state))

    def api(self, method: str, **params: Any) -> Any:
        r = self.tg.post(f"https://api.telegram.org/bot{self.token}/{method}", json=params)
        body = r.json()
        if not body.get("ok"):
            raise RuntimeError(f"telegram {method}: {body.get('description')}")
        return body["result"]

    def send(self, text: str) -> None:
        if self.chat_id:
            for i in range(0, len(text), 3900):
                self.api("sendMessage", chat_id=self.chat_id, text=text[i:i + 3900])

    def _get(self, path: str, **params: Any) -> Any:
        r = self.admin.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        r = self.admin.post(path, json=body)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text

    # ---- commands ----

    def handle(self, text: str) -> str:
        parts = text.strip().split(maxsplit=2)
        cmd = parts[0].split("@")[0].lower() if parts else ""
        arg1 = parts[1] if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""
        if cmd in ("/start", "/help"):
            return HELP
        if cmd == "/status":
            s = self._get("/admin/v1/status")
            nodes = "\n".join(f"- {n['id']} {n['persona']}: {n['balance']} (평판 {n['reputation']}, 예산 {n['budget_state']})"
                              + (f" ⛔ 일시 정지 (/unpause {n['id']})" if n.get("paused") else "")
                              for n in s.get("nodes", []))
            usage = s.get("daily_usage", {}).get("*", 0)
            return f"블록 #{s['height']} · 오늘 LLM {usage}회 · 열린 현상금 {s['open_bounties']}\n{nodes}" + (
                f"\n⚠️ 정지: {s['halted']}" if s.get("halted") else "")
        if cmd == "/approvals":
            items = self._get("/admin/v1/approvals")
            return "\n".join(self._fmt_approval(a) for a in items) or "승인 대기 없음"
        if cmd in ("/approve", "/deny"):
            code, body = self._post(f"/admin/v1/approvals/{arg1}",
                                    {"decision": cmd[1:], "note": rest})
            return f"{arg1} → {body.get('state')}" if code == 200 else f"실패 {code}: {body}"
        if cmd == "/disputes":
            items = self._get("/admin/v1/disputes")
            return "\n".join(f"{d['dispute_id']} {d['target_kind']} {d['opener']}→{d['defendant']}: "
                             f"{(d.get('claim') or {}).get('claim', '')[:150]}" for d in items) or "열린 분쟁 없음"
        if cmd == "/resolve":
            ruling, _, note = rest.partition(" ")
            code, body = self._post(f"/admin/v1/disputes/{arg1}/resolve", {"ruling": ruling, "note": note})
            return f"{arg1} → {body.get('state')}" if code == 200 else f"실패 {code}: {body}"
        if cmd == "/proposals":
            items = self._get("/admin/v1/proposals")
            return "\n".join(f"{x['proposal_id']} [{x['state']}] {x['kind']} '{x['title']}' 표 {x['votes']}"
                             for x in items) or "열린 제안 없음"
        if cmd == "/vote":
            choice, _, reason = rest.partition(" ")
            code, body = self._post(f"/admin/v1/proposals/{arg1}/vote", {"choice": choice, "reason": reason or "-"})
            return f"{arg1} 투표 완료" if code == 200 else f"실패 {code}: {body}"
        if cmd == "/ask":
            if not arg1 or not rest:
                return "사용법: /ask n2 메시지"
            code, body = self._post(f"/admin/v1/nodes/{arg1}/message", {"text": rest})
            return f"{arg1}에게 전달했습니다. 답이 오면 알려드립니다." if code == 200 else f"실패 {code}: {body}"
        if cmd == "/unpause":
            if not arg1:
                return "사용법: /unpause n2 [메모]"
            code, body = self._post(f"/admin/v1/nodes/{arg1}/unpause", {"note": rest})
            return f"{arg1} 일시 정지 해제 (사유였던 것: {body.get('was')})" if code == 200 else f"실패 {code}: {body}"
        if cmd == "/report":
            return f"{self.viewer_url}/reports"
        return "모르는 명령입니다. /help"

    def _fmt_approval(self, a: dict[str, Any]) -> str:
        left = max(0.0, (a["deadline"] - time.time()) / 3600)
        return (f"🔐 {a['approval_id']} · {a['node']} {a['tool']} {a['amount_mP'] / 1000:g}P\n"
                f"{a['reason'][:300]}\n응답 없으면 {left:.0f}시간 뒤 자동 집행 · /approve {a['approval_id']} · "
                f"/deny {a['approval_id']}")

    # ---- loops ----

    def poll_once(self, timeout: int = 25) -> None:
        updates = self.api("getUpdates", offset=self.state["offset"], timeout=timeout)
        for u in updates:
            self.state["offset"] = u["update_id"] + 1
            msg = u.get("message") or {}
            chat = str((msg.get("chat") or {}).get("id", ""))
            text = msg.get("text") or ""
            if not chat or not text:
                continue
            if self.chat_id is None or chat != self.chat_id:
                self.api("sendMessage", chat_id=chat, text=f"이 대화의 chat id: {chat}\n관리인으로 쓰려면 서버 .env의 "
                                                            "TELEGRAM_MANAGER_CHAT_ID에 넣고 서비스를 재시작하세요.")
                continue
            try:
                reply = self.handle(text)
            except Exception as e:  # never crash the bridge on one bad command
                reply = f"오류: {type(e).__name__}: {e}"[:500]
            self.send(reply)
        self._save()

    def push_notifications(self) -> None:
        seen = set(self.state["seen"])
        for a in self._get("/admin/v1/approvals"):
            if a["approval_id"] not in seen:
                self.send("새 승인 요청\n" + self._fmt_approval(a))
                self.state["seen"].append(a["approval_id"])
        for d in self._get("/admin/v1/disputes"):
            if d["dispute_id"] not in seen:
                self.send(f"⚖️ 새 분쟁 {d['dispute_id']} ({d['target_kind']}) {d['opener']}→{d['defendant']}\n"
                          f"{(d.get('claim') or {}).get('claim', '')[:300]}\n/resolve {d['dispute_id']} uphold|reject|void")
                self.state["seen"].append(d["dispute_id"])
        for x in self._get("/admin/v1/proposals"):
            if x["proposal_id"] not in seen:
                self.send(f"🗳 새 제안 {x['proposal_id']} {x['kind']} '{x['title']}'\n"
                          f"/vote {x['proposal_id']} yes|no 이유")
                self.state["seen"].append(x["proposal_id"])
        for m in self._get("/admin/v1/outbox", after=self.state["outbox_after"]):
            self.send(f"💬 {m['node']}: {m['text']}")
            self.state["outbox_after"] = m["id"]
        self._save()

    def run(self, push_every: float = 30.0) -> None:
        last = 0.0
        while True:
            try:
                self.poll_once()
                if time.monotonic() - last >= push_every:
                    last = time.monotonic()
                    self.push_notifications()
            except Exception as e:
                print(f"[telegram] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                time.sleep(10)


def main(argv: list[str] | None = None) -> int:
    from aichain.sim.factory import load_default_env

    ap = argparse.ArgumentParser(description="AIChain Telegram bridge for the node manager")
    ap.add_argument("--admin-url", default="http://127.0.0.1:8780")
    ap.add_argument("--token-file", type=Path, default=ROOT / "data" / "admin_token")
    ap.add_argument("--viewer-url", default="http://100.98.150.35:8790")
    args = ap.parse_args(argv)
    load_default_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set (create a bot with @BotFather, put the token in .env)", file=sys.stderr)
        return 2
    admin = httpx.Client(base_url=args.admin_url, timeout=60,
                         headers={"Authorization": f"Bearer {args.token_file.read_text().strip()}"})
    TelegramBridge(token, os.environ.get("TELEGRAM_MANAGER_CHAT_ID") or None, admin,
                   state_path=ROOT / "data" / "telegram_state.json", viewer_url=args.viewer_url).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
