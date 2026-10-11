"""Deterministic fake-model brains for the scripted adapter.

`persona_brain` reads the event and persona from the prompt and acts the way each
persona's layer describes, in one model call (action + done). Good enough to drive
every protocol path in tests without a real model.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from aichain.gateways.llm.base import LLMRequest, LLMResponse
from aichain.gateways.llm.scripted import tool_response

_EVENT = re.compile(r'<event type="(\w+)"([^>]*)>')
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_DATA = re.compile(r"<data [^>]*>\n(.*?)\n</data>", re.S)
_ANSWER = re.compile(r'<answer id="([^"]+)" author="([^"]+)">\n<data [^>]*>\n(.*?)\n</data>', re.S)


def persona_of(request: LLMRequest) -> str:
    system = request.messages[0].content or ""
    for name in ("Seeker", "Artisan", "Warden", "Nomad"):
        if f"나는 {name}다" in system or f"나는 {name}이다" in system:
            return name.lower()
    return "unknown"


def event_of(request: LLMRequest) -> tuple[str, dict[str, str], str]:
    user = next(m.content for m in reversed(request.messages) if m.role == "user") or ""
    m = _EVENT.search(user)
    if not m:
        return "unknown", {}, user
    return m.group(1), dict(_ATTR.findall(m.group(2))), user


def _json_data(text: str) -> dict[str, Any]:
    m = _DATA.search(text)
    return json.loads(html.unescape(m.group(1))) if m else {}


def persona_brain(request: LLMRequest, *, judge: bool = True) -> LLMResponse:
    # A tool result after the last user turn means our action already ran: finish.
    if request.messages[-1].role == "tool":
        return tool_response(("done", {"summary": "처리 완료"}))
    persona = persona_of(request)
    kind, attrs, text = event_of(request)

    if kind == "bridge_question" and persona == "seeker":
        q = _json_data(text)
        return tool_response(
            ("ask", {"question": q, "reward_P": 20, "tags": ["code"]}),
            ("done", {"summary": "브릿지 질문에 20P 현상금을 걸었다."}),
        )
    if kind == "bounty_new":
        tags = attrs.get("tags", "").split(",")
        if (persona == "artisan" and "code" in tags) or persona == "nomad":
            confidence = 0.9 if persona == "artisan" else 0.6
            return tool_response(
                ("answer", {"bounty_id": attrs["bounty_id"], "answer": {
                    "answer": f"{persona}의 답: 리스트를 뒤집으려면 lst[::-1]을 쓴다.",
                    "confidence": confidence,
                    "sources": [],
                    "verification": "assert [1,2,3][::-1] == [3,2,1]",
                }}),
                ("done", {"summary": f"{attrs['bounty_id']}에 답했다."}),
            )
        return tool_response(("done", {"summary": "내 분야가 아니라 지나쳤다."}))
    if kind == "judge_due" and judge:
        answers = []
        for aid, author, body in _ANSWER.findall(text):
            answers.append((json.loads(html.unescape(body)).get("confidence", 0), aid, author))
        if answers:
            best = max(answers)
            return tool_response(
                ("judge", {"bounty_id": attrs["bounty_id"], "accept": best[1]}),
                ("remember", {"kind": "relation", "content": f"{best[2]}의 답변을 수락했다.",
                              "tags": [best[2]]}),
                ("done", {"summary": f"{best[2]}의 답을 수락했다."}),
            )
    return tool_response(("done", {"summary": "할 일 없음"}))


def lazy_asker_brain(request: LLMRequest) -> LLMResponse:
    """Like persona_brain, but never judges — exercises AUTO_SPLIT."""
    return persona_brain(request, judge=False)
