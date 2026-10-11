"""Search Gateway client (design doc 05, ch. 1): per-node limits and caches in front of
the headless browser service.
"""

from __future__ import annotations

import re
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
    search_per_hour: int = 6  # paid search API: keep it small
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


class BraveSearchProvider:
    """Brave Search API (paid per query): web search only. Pages are still read by the
    isolated browser. A daily cap persisted on disk keeps spend bounded across restarts."""

    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, daily_cap: int = 60, usage_path: str | None = None,
                 min_interval: float = 1.1, client: httpx.Client | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        if not api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY is empty")
        self._key = api_key
        self.daily_cap = daily_cap
        self.usage_path = usage_path
        self.min_interval = min_interval
        self._client = client or httpx.Client(timeout=20)
        self._clock = clock
        self._last = 0.0
        self._day, self.used_today = self._load()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self._clock()))

    def _load(self) -> tuple[str, int]:
        import json
        import os

        if self.usage_path and os.path.exists(self.usage_path):
            with open(self.usage_path, encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("day") == self._today():
                return raw["day"], int(raw.get("count", 0))
        return self._today(), 0

    def _save(self) -> None:
        import json
        import os

        if self.usage_path:
            tmp = self.usage_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"day": self._day, "count": self.used_today}, f)
            os.replace(tmp, self.usage_path)

    def search(self, query: str, k: int) -> dict[str, Any]:
        if self._today() != self._day:
            self._day, self.used_today = self._today(), 0
        if self.used_today >= self.daily_cap:
            raise WebError(f"Brave search daily cap {self.daily_cap} reached (resets 00:00 UTC)")
        wait = self.min_interval - (self._clock() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = self._clock()
        started = time.monotonic()
        self.used_today += 1  # count before the call: a failed request may still be billed
        self._save()
        try:
            r = self._client.get(self.URL, params={"q": query, "count": k},
                                 headers={"X-Subscription-Token": self._key,
                                          "Accept": "application/json"})
        except httpx.HTTPError as e:
            raise WebError(f"Brave search unreachable: {e!r}") from e
        if r.status_code >= 400:
            raise WebError(f"Brave search HTTP {r.status_code}: {r.text[:200]}",
                           blocked=r.status_code in (401, 403, 429))
        hits = (r.json().get("web") or {}).get("results") or []
        results = [{"title": h.get("title", ""), "url": h.get("url", ""),
                    "snippet": re.sub(r"<[^>]+>", "", h.get("description", ""))}
                   for h in hits[:k] if h.get("url", "").startswith(("http://", "https://"))]
        return {"query": query, "results": results, "engine": "brave-api",
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "used_today": self.used_today, "daily_cap": self.daily_cap}


class CompositeProvider:
    """Search from one provider (e.g. Brave API), fetch from another (the isolated browser)."""

    def __init__(self, search: WebProvider, fetch: WebProvider) -> None:
        self._search, self._fetch = search, fetch

    def search(self, query: str, k: int) -> dict[str, Any]:
        return self._search.search(query, k)

    def fetch(self, url: str, max_chars: int) -> dict[str, Any]:
        return self._fetch.fetch(url, max_chars)
