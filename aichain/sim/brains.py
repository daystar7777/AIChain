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


_URL = re.compile(r"^\s+(https?://\S+)$", re.M)


def research_brain(request: LLMRequest) -> LLMResponse:
    """Plays a research episode: search → fetch the first result → remember → note → done."""
    kind, _, text = event_of(request)
    if kind != "research_tick":
        return persona_brain(request)
    tool_msgs = [m for m in request.messages if m.role == "tool"]
    topic = re.search(r"## 관심사\n- ([^:]+):", text)
    topic = topic.group(1).strip() if topic else "주제"
    if not tool_msgs:
        return tool_response(("search", {"query": f"{topic} 최신", "k": 3}))
    last = tool_msgs[-1]
    if last.name == "search":
        urls = _URL.findall(html.unescape(last.content or ""))
        if not urls:
            return tool_response(("done", {"summary": "검색 결과 없음"}))
        return tool_response(("fetch", {"url": urls[0], "max_chars": 2000}))
    if last.name == "fetch":
        m = re.search(r'<data source="web:([^"]+)"', last.content or "")
        url = html.unescape(m.group(1)) if m else ""
        body = html.unescape((last.content or "").split("\n\n", 1)[-1])[:80]
        return tool_response(
            ("remember", {"kind": "fact", "content": f"{topic}: {body}", "tags": [topic],
                          "source_url": url}),
            ("note", {"topic": topic, "summary": f"{topic}에 대해 읽은 내용 요약", "confidence": 0.5,
                      "facts": [{"text": body, "source_url": url}],
                      "open_questions": ["더 최신 자료 확인"]}),
            ("done", {"summary": f"{topic} 연구 노트를 갱신했다."}),
        )
    return tool_response(("done", {"summary": "연구 종료"}))
