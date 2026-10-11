"""LLM Gateway core: provider-neutral messages, the ProviderAdapter interface,
per-node token buckets and retry with backoff (design docs 03 ch. 5, 04 ch. 2 and 6).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str  # raw JSON text as produced by the model


@dataclass
class Message:
    """One conversation item.

    role: system | user | assistant | tool
    For assistant messages, `raw` keeps the provider's own output items so they can be
    replayed verbatim on the next request (needed by the Responses API).
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    call_id: str | None = None  # for role == "tool"
    name: str | None = None  # for role == "tool"
    raw: list[dict[str, Any]] | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LLMRequest:
    node_id: str
    model: str
    messages: list[Message]
    tools: list[ToolSpec]
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    usage: Usage
    raw: list[dict[str, Any]] | None = None


class ProviderError(Exception):
    def __init__(self, message: str, *, status: int | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        # None: network/timeout. 0: a bug in our adapter (not worth retrying).
        if self.status is None:
            return True
        return self.status == 429 or self.status >= 500


class ProviderAdapter(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class BudgetState(StrEnum):
    ACTIVE = "ACTIVE"
    CONSERVE = "CONSERVE"
    SLEEP = "SLEEP"


class BudgetExhausted(Exception):
    pass


class TokenBucket:
    """Per-node LLM budget (03 doc 5.2) with the state machine of 04 doc ch. 6.

    capacity = 0.5·D, refill = D/1440 per minute, reserve = 10% of capacity for duties.
    Time comes from `clock()` in minutes so simulations can use a virtual clock.
    """

    def __init__(self, daily_tokens: int, clock: Callable[[], float]) -> None:
        self.daily = daily_tokens
        self.capacity = daily_tokens // 2
        self.refill_per_min = daily_tokens / 1440
        self.reserve = self.capacity // 10
        self.tokens = float(self.capacity)
        self._clock = clock
        self._last = clock()
        self.state = BudgetState.ACTIVE

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.refill_per_min)
        self._last = now
        self._update_state()

    def _update_state(self) -> None:
        t, cap = self.tokens, self.capacity
        if self.state == BudgetState.ACTIVE and t < 0.25 * cap:
            self.state = BudgetState.CONSERVE
        if self.state == BudgetState.CONSERVE:
            if t < self.reserve:
                self.state = BudgetState.SLEEP
            elif t >= 0.5 * cap:
                self.state = BudgetState.ACTIVE
        elif self.state == BudgetState.SLEEP and t >= 0.25 * cap:
            self.state = BudgetState.CONSERVE

    def remaining(self) -> int:
        self._refill()
        return int(self.tokens)

    def current_state(self) -> BudgetState:
        self._refill()
        return self.state

    def allows(self, priority: int) -> bool:
        """Events processed per state: ACTIVE all, CONSERVE 1–3, SLEEP only duties (1)."""
        state = self.current_state()
        if state == BudgetState.ACTIVE:
            return True
        if state == BudgetState.CONSERVE:
            return priority <= 3
        return priority == 1 and self.tokens > 0

    def can_spend(self, priority: int) -> bool:
        self._refill()
        floor = 0 if priority == 1 else self.reserve
        return self.tokens > floor

    def charge(self, weighted_tokens: float) -> None:
        self._refill()
        self.tokens = max(0.0, self.tokens - weighted_tokens)
        self._update_state()


class CallLimitReached(Exception):
    pass


@dataclass
class GatewayConfig:
    output_weight: float = 2.0  # output token cost relative to input (Muse Spark: 0.2 / 0.1)
    max_retries: int = 5
    max_calls: int | None = None  # hard cap on total provider calls (smoke tests)
    daily_calls_per_node: int | None = None  # per UTC day
    daily_calls_total: int | None = None  # per UTC day, all nodes
    # Persist daily counters here so restarts cannot reset the caps.
    usage_path: str | None = None


class LLMGateway:
    """Relays every model call, enforces budgets, retries transient failures.

    Nodes never hold provider keys; adapters do (03 doc 5.1).
    """

    def __init__(
        self,
        adapters: dict[str, ProviderAdapter],
        config: GatewayConfig | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.adapters = adapters
        self.config = config or GatewayConfig()
        self.buckets: dict[str, TokenBucket] = {}
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self.calls = 0
        self._daily: dict[tuple[str, str], int] = {}  # (utc date, node | "*") -> calls
        self._load_usage()
        self.log: list[dict[str, Any]] = []
        # Called after every provider call: (request, response | None, info). The journal
        # uses this to record full prompts and responses.
        self.observers: list[Callable[[LLMRequest, LLMResponse | None, dict[str, Any]], None]] = []

    def register(self, node_id: str, bucket: TokenBucket) -> None:
        self.buckets[node_id] = bucket

    def chat(self, provider: str, request: LLMRequest, priority: int) -> LLMResponse:
        bucket = self.buckets[request.node_id]
        if not bucket.can_spend(priority):
            raise BudgetExhausted(f"{request.node_id}: budget {bucket.state}")
        if self.config.max_calls is not None and self.calls >= self.config.max_calls:
            raise CallLimitReached(f"gateway call cap {self.config.max_calls} reached")
        day = time.strftime("%Y-%m-%d", time.gmtime())
        node_cap, total_cap = self.config.daily_calls_per_node, self.config.daily_calls_total
        if node_cap is not None and self._daily.get((day, request.node_id), 0) >= node_cap:
            raise CallLimitReached(f"{request.node_id}: daily call cap {node_cap} reached")
        if total_cap is not None and self._daily.get((day, "*"), 0) >= total_cap:
            raise CallLimitReached(f"daily call cap {total_cap} (all nodes) reached")
        self._daily[(day, request.node_id)] = self._daily.get((day, request.node_id), 0) + 1
        self._daily[(day, "*")] = self._daily.get((day, "*"), 0) + 1
        self._save_usage(day)
        adapter = self.adapters[provider]
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                self.calls += 1
                resp = adapter.complete(request)
                break
            except ProviderError as e:
                self._notify(request, None, {"provider": provider, "attempt": attempt,
                                             "latency_ms": _ms(started), "error": str(e),
                                             "status": e.status})
                if not e.retryable or attempt >= self.config.max_retries:
                    raise
                delay = e.retry_after if e.retry_after is not None else (
                    2**attempt + self._rng.uniform(0, 0.5))
                attempt += 1
                self._sleep(min(delay, 60.0))
            except Exception as e:  # adapter bug or malformed data: non-retryable provider error
                err = ProviderError(f"adapter error: {type(e).__name__}: {e}"[:300], status=0)
                self._notify(request, None, {"provider": provider, "attempt": attempt,
                                             "latency_ms": _ms(started), "error": str(err), "status": 0})
                raise err from e
        latency_ms = _ms(started)
        # Only successful calls are charged.
        weighted = resp.usage.input_tokens + self.config.output_weight * resp.usage.output_tokens
        bucket.charge(weighted)
        self.log.append({
            "node": request.node_id, "provider": provider, "model": request.model,
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
            "tool_calls": [c.name for c in resp.tool_calls], "retries": attempt,
        })
        self._notify(request, resp, {"provider": provider, "attempt": attempt,
                                     "latency_ms": latency_ms,
                                     "weighted_tokens": weighted,
                                     "budget_remaining": bucket.remaining(),
                                     "budget_state": str(bucket.state)})
        return resp

    def _load_usage(self) -> None:
        import json
        import os

        path = self.config.usage_path
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self._daily = {(raw["day"], k): v for k, v in raw.get("calls", {}).items()}

    def _save_usage(self, day: str) -> None:
        import json
        import os

        path = self.config.usage_path
        if not path:
            return
        calls = {k: v for (d, k), v in self._daily.items() if d == day}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"day": day, "calls": calls}, f)
        os.replace(tmp, path)

    def has_daily_room(self, node_id: str) -> bool:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        node_cap, total_cap = self.config.daily_calls_per_node, self.config.daily_calls_total
        if node_cap is not None and self._daily.get((day, node_id), 0) >= node_cap:
            return False
        return not (total_cap is not None and self._daily.get((day, "*"), 0) >= total_cap)

    def daily_usage(self) -> dict[str, int]:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        return {k: v for (d, k), v in self._daily.items() if d == day}

    def _notify(self, request: LLMRequest, resp: LLMResponse | None, info: dict[str, Any]) -> None:
        for observer in self.observers:
            observer(request, resp, info)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
