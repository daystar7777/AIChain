"""Scripted adapter: a fake model that emits predetermined tool calls (08 doc 6, simulation).

A *brain* is a function from the request to a response, so tests can play out any
protocol path deterministically and without network access.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable

from aichain.gateways.llm.base import LLMRequest, LLMResponse, ToolCall, Usage

Brain = Callable[[LLMRequest], LLMResponse]


def tool_response(*calls: tuple[str, dict[str, Any]], text: str | None = None,
                  usage: Usage | None = None) -> LLMResponse:
    tool_calls = [
        ToolCall(call_id=f"call_{i}_{name}", name=name,
                 arguments=json.dumps(args, ensure_ascii=False))
        for i, (name, args) in enumerate(calls)
    ]
    return LLMResponse(text=text, tool_calls=tool_calls,
                       usage=usage or Usage(input_tokens=1200, output_tokens=150))


class ScriptedAdapter:
    name = "scripted"

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        # Snapshot: the caller keeps appending to the same message list.
        request = replace(request, messages=list(request.messages))
        self.requests.append(request)
        return self.brain(request)
