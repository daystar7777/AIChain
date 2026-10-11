"""Search Gateway client (design doc 05, ch. 1): per-node limits and caches in front of
the headless browser service.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import httpx


class WebError(Exception):
    def __init__(self, message: str, *, blocked: bool = False) -> None:
        super().__init__(message)
        self.blocked = blocked


class WebProvider(Protocol):
    def search(self, query: str, k: int) -> dict[str, Any]: ...

    def fetch(self, url: str, max_chars: int) -> dict[str, Any]: ...


class BrowserServiceProvider:
    """Talks to `aichain.browser.service` on loopback."""

    def __init__(self, base_url: str = "http://127.0.0.1:8781", timeout: float = 60.0) -> None:
        if base_url.startswith("unix:"):  # e.g. unix:/run/aichain-browser/browser.sock
            transport = httpx.HTTPTransport(uds=base_url[5:])
            self._client = httpx.Client(base_url="http://browser", transport=transport, timeout=timeout)
        else:
            self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self._client.post(path, json=body)
        except httpx.HTTPError as e:
            raise WebError(f"browser service unreachable: {e!r}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise WebError(str(detail)[:300], blocked=r.status_code in (403, 429))
        return r.json()

    def search(self, query: str, k: int) -> dict[str, Any]:
        return self._post("/search", {"query": query, "k": k})

    def fetch(self, url: str, max_chars: int) -> dict[str, Any]:
        return self._post("/fetch", {"url": url, "max_chars": max_chars})


@dataclass
class WebLimits:
    search_per_hour: int = 20
    fetch_per_hour: int = 40
    search_ttl: float = 3600
    fetch_ttl: float = 86400


class SearchGateway:
    def __init__(self, provider: WebProvider, limits: WebLimits | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.provider = provider
        self.limits = limits or WebLimits()
        self._clock = clock
        self._history: dict[tuple[str, str], deque[float]] = {}
        # Caches are per node: one node's results never leak into another's (05 doc 1.3).
        self._cache: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}

    def _take(self, node: str, kind: str, limit: int) -> None:
        now = self._clock()
        q = self._history.setdefault((node, kind), deque())
        while q and q[0] <= now - 3600:
            q.popleft()
        if len(q) >= limit:
            raise WebError(f"{kind} limit {limit}/hour reached for {node}")
        q.append(now)

    def _cached(self, node: str, kind: str, key: str, ttl: float) -> dict[str, Any] | None:
        hit = self._cache.get((node, kind, key))
        if hit and self._clock() - hit[0] < ttl:
            return {**hit[1], "cached": True}
        return None

    def search(self, node: str, query: str, k: int) -> dict[str, Any]:
        key = f"{k}|{query.strip().lower()}"
        if (hit := self._cached(node, "search", key, self.limits.search_ttl)) is not None:
            return hit
        self._take(node, "search", self.limits.search_per_hour)
        result = self.provider.search(query, k)
        self._cache[(node, "search", key)] = (self._clock(), result)
        return result

    def fetch(self, node: str, url: str, max_chars: int) -> dict[str, Any]:
        key = f"{max_chars}|{url}"
        if (hit := self._cached(node, "fetch", key, self.limits.fetch_ttl)) is not None:
            return hit
        self._take(node, "fetch", self.limits.fetch_per_hour)
        result = self.provider.fetch(url, max_chars)
        self._cache[(node, "fetch", key)] = (self._clock(), result)
        return result


class FakeWebProvider:
    """In-memory web for tests: {query: results} and {url: page}."""

    def __init__(self, results: dict[str, list[dict[str, str]]] | None = None,
                 pages: dict[str, dict[str, str]] | None = None) -> None:
        self.results = results or {}
        self.pages = pages or {}
        self.calls: list[tuple[str, str]] = []

    def search(self, query: str, k: int) -> dict[str, Any]:
        self.calls.append(("search", query))
        hits = next((v for q, v in self.results.items() if q in query), [])
        return {"query": query, "results": hits[:k], "elapsed_ms": 1, "engine": "fake"}

    def fetch(self, url: str, max_chars: int) -> dict[str, Any]:
        self.calls.append(("fetch", url))
        page = self.pages.get(url)
        if page is None:
            raise WebError(f"HTTP 404 from {url}")
        text = page["text"][:max_chars]
        return {"url": url, "final_url": url, "title": page.get("title", ""), "text": text,
                "chars": len(text), "fetched_at": "2026-10-11T00:00:00+00:00", "elapsed_ms": 1,
                "status": 200, "blocked_requests": []}
