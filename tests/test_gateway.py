import json

import httpx
import pytest

from aichain.gateways.llm.base import (
    BudgetExhausted,
    BudgetState,
    CallLimitReached,
    GatewayConfig,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderError,
    TokenBucket,
    ToolCall,
    ToolSpec,
    Usage,
)
from aichain.gateways.llm.opencode_go import OpenCodeGoAdapter, to_input_items


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_bucket_state_machine_and_refill():
    clock = Clock()
    b = TokenBucket(1440_000, clock)  # capacity 720k, refill 1000/min, reserve 72k
    assert b.current_state() == BudgetState.ACTIVE
    b.charge(560_000)  # 160k left < 25%
    assert b.current_state() == BudgetState.CONSERVE
    assert b.allows(3) and not b.allows(4)
    b.charge(100_000)  # 60k < reserve
    assert b.current_state() == BudgetState.SLEEP
    assert b.allows(1) and not b.allows(2)
    assert not b.can_spend(3) and b.can_spend(1)
    clock.now = 120  # +120k → 180k ≥ 25%
    assert b.current_state() == BudgetState.CONSERVE
    clock.now = 300  # 360k ≥ 50%
    assert b.current_state() == BudgetState.ACTIVE
    clock.now = 10_000
    assert b.remaining() == b.capacity


class FlakyAdapter:
    name = "flaky"

    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return LLMResponse("ok", [], Usage(100, 50))


def gateway_with(adapter, **cfg):
    sleeps = []
    gw = LLMGateway({"p": adapter}, GatewayConfig(**cfg), sleep=sleeps.append)
    gw.register("n1", TokenBucket(2_000_000, Clock()))
    return gw, sleeps


def req():
    return LLMRequest("n1", "m", [Message("user", "hi")], [])


def test_retries_transient_errors_with_backoff_and_retry_after():
    adapter = FlakyAdapter([ProviderError("busy", status=429, retry_after=3.0),
                            ProviderError("boom", status=503)])
    gw, sleeps = gateway_with(adapter)
    resp = gw.chat("p", req(), priority=3)
    assert resp.text == "ok" and adapter.calls == 3
    assert sleeps[0] == 3.0 and 2 <= sleeps[1] < 2.5
    # Only the successful call is charged: 100 + 2×50.
    assert gw.buckets["n1"].remaining() == 1_000_000 - 200


def test_non_retryable_error_raises_immediately():
    adapter = FlakyAdapter([ProviderError("bad request", status=400)])
    gw, sleeps = gateway_with(adapter)
    with pytest.raises(ProviderError):
        gw.chat("p", req(), priority=3)
    assert adapter.calls == 1 and sleeps == []


def test_gives_up_after_max_retries():
    adapter = FlakyAdapter([ProviderError("x", status=500)] * 10)
    gw, _ = gateway_with(adapter, max_retries=2)
    with pytest.raises(ProviderError):
        gw.chat("p", req(), priority=3)
    assert adapter.calls == 3


def test_call_cap_and_budget_gate():
    gw, _ = gateway_with(FlakyAdapter([]), max_calls=1)
    gw.chat("p", req(), priority=3)
    with pytest.raises(CallLimitReached):
        gw.chat("p", req(), priority=3)
    gw2, _ = gateway_with(FlakyAdapter([]))
    gw2.buckets["n1"].charge(10**9)
    with pytest.raises(BudgetExhausted):
        gw2.chat("p", req(), priority=3)


def test_opencode_go_request_and_response_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "done",
                 "arguments": "{\"summary\": \"ok\"}"},
            ],
            "usage": {"input_tokens": 321, "output_tokens": 12},
        })

    adapter = OpenCodeGoAdapter("test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    request = LLMRequest(
        "n2", "muse-spark-1.3-contributor",
        [Message("system", "core"), Message("user", "<event/>")],
        [ToolSpec("done", "end", {"type": "object", "properties": {}})],
        temperature=0.4, max_output_tokens=100,
    )
    resp = adapter.complete(request)
    assert seen["url"] == "https://opencode.ai/zen/go/v1/responses"
    assert seen["headers"]["authorization"] == "Bearer test-key"
    assert seen["headers"]["x-opencode-session"].startswith("aichain-n2-")
    assert seen["headers"]["user-agent"].startswith("aichain-proto/")
    assert seen["body"]["model"] == "muse-spark-1.3-contributor"
    assert seen["body"]["tools"][0] == {"type": "function", "name": "done", "description": "end",
                                        "parameters": {"type": "object", "properties": {}}}
    assert resp.tool_calls == [ToolCall("call_1", "done", "{\"summary\": \"ok\"}")]
    assert resp.usage == Usage(321, 12)

    # Next turn replays the raw output items, then the tool output.
    items = to_input_items([Message("assistant", None, resp.tool_calls, raw=resp.raw),
                            Message("tool", "fine", call_id="call_1", name="done")])
    assert items[0]["type"] == "reasoning" and items[1]["type"] == "function_call"
    assert items[2] == {"type": "function_call_output", "call_id": "call_1", "output": "fine"}


def test_opencode_go_maps_http_errors():
    def handler(request):
        return httpx.Response(429, headers={"retry-after": "7"}, text="slow down")

    adapter = OpenCodeGoAdapter("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError) as e:
        adapter.complete(req())
    assert e.value.status == 429 and e.value.retry_after == 7.0 and e.value.retryable
