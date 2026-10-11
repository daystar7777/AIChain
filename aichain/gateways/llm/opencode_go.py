"""OpenCode Go adapter (OpenAI Responses API format).

Muse Spark 1.3 is served as `muse-spark-1.3-contributor` at
https://opencode.ai/zen/go/v1/responses. The key comes from OPENCODE_GO_API_KEY.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import Any

import httpx

from aichain.gateways.llm.base import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "muse-spark-1.3-contributor"
USER_AGENT = "aichain-proto/0.0.1 (+https://github.com/daystar7777/AIChain)"


def to_input_items(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role in ("system", "user"):
            items.append({"role": m.role, "content": m.content or ""})
        elif m.role == "assistant":
            if m.raw is not None:
                items.extend(m.raw)
                continue
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            for c in m.tool_calls:
                items.append({"type": "function_call", "call_id": c.call_id,
                              "name": c.name, "arguments": c.arguments})
        elif m.role == "tool":
            items.append({"type": "function_call_output", "call_id": m.call_id,
                          "output": m.content or ""})
        else:
            raise ValueError(f"unknown role {m.role}")
    return items


def parse_output(body: dict[str, Any]) -> LLMResponse:
    if not isinstance(body, dict):
        raise ProviderError("response is not a JSON object", status=502)
    if body.get("error"):
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise ProviderError(f"provider error: {msg}"[:300], status=502)
    status = body.get("status")
    if status == "failed":
        raise ProviderError("response status failed", status=502)
    output = body.get("output") or []
    if not isinstance(output, list):
        raise ProviderError("response output is not a list", status=502)
    texts: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{len(calls)}"
            args = item.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args if args is not None else {}, ensure_ascii=False)
            calls.append(ToolCall(call_id, str(item.get("name", "")), args))
        elif kind == "message":
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    texts.append(part["text"])
    if status == "incomplete" and not calls and not texts:
        reason = (body.get("incomplete_details") or {}).get("reason", "unknown")
        raise ProviderError(f"response incomplete ({reason}) with no usable output", status=422)
    usage = body.get("usage") or {}
    return LLMResponse(
        text="\n".join(texts) or None,
        tool_calls=calls,
        usage=Usage(int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)),
        raw=output,
    )


class OpenCodeGoAdapter:
    name = "opencode-go"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENCODE_GO_API_KEY is empty")
        self._key = api_key
        self._url = base_url.rstrip("/") + "/responses"
        self._client = client or httpx.Client(timeout=timeout)
        # One stable session id per node, as OpenCode Go recommends.
        self._sessions: dict[str, str] = {}

    def _session(self, node_id: str) -> str:
        return self._sessions.setdefault(node_id, f"aichain-{node_id}-{uuid.uuid4().hex[:12]}")

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": to_input_items(request.messages),
        }
        if request.tools:
            payload["tools"] = [
                {"type": "function", "name": t.name, "description": t.description,
                 "parameters": t.parameters}
                for t in request.tools
            ]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            return self._complete(request, replay_raw=True)
        except ProviderError as e:
            # Some backends reject replayed output items (e.g. reasoning items without stored
            # state). Retry once with plain function_call items instead.
            if e.status == 400 and any(m.raw for m in request.messages):
                return self._complete(request, replay_raw=False)
            raise

    def _complete(self, request: LLMRequest, replay_raw: bool) -> LLMResponse:
        if not replay_raw:
            request = replace(request, messages=[replace(m, raw=None) for m in request.messages])
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "x-opencode-session": self._session(request.node_id),
        }
        try:
            r = self._client.post(self._url, headers=headers, json=self.build_payload(request))
        except httpx.TimeoutException as e:
            raise ProviderError(f"timeout: {e!r}") from e  # retryable (status None)
        except httpx.HTTPError as e:
            raise ProviderError(f"network error: {e!r}") from e
        if r.status_code >= 400:
            retry_after = r.headers.get("retry-after")
            raise ProviderError(
                f"HTTP {r.status_code}: {r.text[:500]}",
                status=r.status_code,
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        try:
            body = r.json()
        except json.JSONDecodeError as e:
            raise ProviderError(f"invalid JSON response: {r.text[:200]}", status=r.status_code) from e
        return parse_output(body)
