"""Headless browser service: `search` and `fetch` over HTTP for the node processes.

    python -m aichain.browser.service --port 8781

Runs as its own unprivileged user (systemd unit aichain-browser) so the browser
cannot read keys, the journal or .env. Chromium keeps its sandbox and reaches the
network only through `GuardProxy`. One navigation at a time, with a minimum gap
between navigations shared by all nodes.
"""

import argparse
import asyncio
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import json

from aichain.browser.ddg import (
    SearchBlocked,
    parse_results,
    parse_wikipedia,
    search_url,
    wikipedia_search_url,
)
from aichain.browser.guard import Blocked, EgressGuard
from aichain.browser.proxy import GuardProxy, UnixForwarder

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36 AIChain-research/0.0.1")

EXTRACT_JS = """() => {
  document.querySelectorAll('script,style,noscript,template,iframe,svg,canvas,[aria-hidden="true"],[hidden]')
    .forEach(e => e.remove());
  const hidden = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0'
        || parseFloat(cs.fontSize) < 2) hidden.push(el);
  }
  hidden.forEach(e => e.remove());
  const root = document.querySelector('article') || document.querySelector('main') || document.body;
  return { title: document.title || '', text: root ? root.innerText : '' };
}"""


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    k: int = Field(default=8, ge=1, le=8)
    region: str | None = Field(default=None, max_length=10)


class FetchIn(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    max_chars: int = Field(default=20_000, ge=500, le=20_000)


def clean_text(text: str, max_chars: int) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text[:max_chars]


class Browser:
    """`egress_uds`: production mode — the browser runs in a private network namespace and
    reaches the web only through the egress service's Unix socket. Without it (development),
    an in-process GuardProxy is used."""

    def __init__(self, guard: EgressGuard, min_interval: float = 5.0, timeout_ms: int = 20_000,
                 egress_uds: str | None = None, chromium_sandbox: bool = True,
                 skip_ddg: bool = False) -> None:
        self.guard = guard
        self.skip_ddg = skip_ddg
        self.min_interval = min_interval
        self.timeout_ms = timeout_ms
        self.chromium_sandbox = chromium_sandbox
        self.blocked: list[dict[str, str]] = []
        self.proxy = UnixForwarder(egress_uds) if egress_uds else GuardProxy(guard, on_block=self._blocked)
        self.egress_uds = egress_uds
        self._lock = asyncio.Lock()
        self._last = 0.0
        self._pw = None
        self._browser = None

    def _blocked(self, target: str, reason: str) -> None:
        self.blocked.append({"target": target, "reason": reason})

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        port = await self.proxy.start()
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            chromium_sandbox=self.chromium_sandbox,
            proxy={"server": f"http://127.0.0.1:{port}"},
            args=[
                "--proxy-bypass-list=<-loopback>",  # send loopback through the proxy too (→ blocked)
                "--disable-quic",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--disable-background-networking",
            ],
        )

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        await self.proxy.close()

    async def _navigate(self, url: str, extract: bool) -> dict[str, Any]:
        # Check the top-level URL up front for a clear error; the proxy re-checks everything.
        if self.egress_uds:
            self.guard.precheck(url)  # no DNS in the private namespace; egress resolves and checks
        else:
            await self.guard.check_url(url)
        async with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
            started = time.monotonic()
            mark = len(self.blocked)
            context = await self._browser.new_context(
                user_agent=USER_AGENT, accept_downloads=False, java_script_enabled=True,
                service_workers="block", locale="ko-KR")

            async def route(r):
                if r.request.resource_type in ("image", "media", "font"):
                    await r.abort()
                else:
                    await r.continue_()

            await context.route("**/*", route)
            page = await context.new_page()
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                status = resp.status if resp else None
                if resp is not None and (await resp.header_value("x-aichain-blocked")) == "True":
                    raise Blocked(f"egress proxy refused {url}")
                if extract:
                    with_text = await page.evaluate(EXTRACT_JS)
                else:
                    with_text = {"title": await page.title(), "html": await page.content()}
                final_url = page.url
            finally:
                await context.close()
            return {"status": status, "final_url": final_url,
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "blocked_requests": self.blocked[mark:], **with_text}

    async def search(self, query: str, k: int, region: str | None) -> dict[str, Any]:
        notes: list[str] = []
        if not self.skip_ddg:
            try:
                page = await self._navigate(search_url(query, region), extract=False)
                if page["status"] != 200:
                    # DuckDuckGo answers bot checks with 202/403. Reported, never bypassed.
                    raise SearchBlocked(f"duckduckgo answered HTTP {page['status']} (bot check)")
                results = parse_results(page.pop("html"), k)
                return {"query": query, "results": results, "elapsed_ms": page["elapsed_ms"],
                        "engine": "duckduckgo-html"}
            except SearchBlocked as e:
                notes.append(str(e))
        # Fallback: Wikipedia's official search API (covers encyclopedic topics only).
        lang, url = wikipedia_search_url(query, k)
        page = await self._navigate(url, extract=True)
        if page["status"] != 200:
            raise SearchBlocked("; ".join(notes + [f"wikipedia answered HTTP {page['status']}"]))
        results = parse_wikipedia(lang, json.loads(page["text"]), k)
        return {"query": query, "results": results, "elapsed_ms": page["elapsed_ms"],
                "engine": f"wikipedia-{lang}", "notes": notes}

    async def fetch(self, url: str, max_chars: int) -> dict[str, Any]:
        page = await self._navigate(url, extract=True)
        if page["status"] and page["status"] >= 400:
            raise HTTPException(502, f"HTTP {page['status']} from {url}")
        text = clean_text(page.pop("text"), max_chars)
        return {"url": url, "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "chars": len(text), "text": text, **page}


def create_app(browser: Browser, start: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if start:
            await browser.start()
        try:
            yield
        finally:
            if start:
                await browser.close()

    app = FastAPI(title="AIChain browser", version="0.0.1", lifespan=lifespan,
                  docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "blocked_total": len(browser.blocked)}

    @app.post("/search")
    async def search(q: SearchIn) -> dict[str, Any]:
        try:
            return await browser.search(q.query, q.k, q.region)
        except SearchBlocked as e:
            raise HTTPException(429, str(e)) from e
        except Blocked as e:
            raise HTTPException(403, f"blocked: {e}") from e

    @app.post("/fetch")
    async def fetch(f: FetchIn) -> dict[str, Any]:
        try:
            return await browser.fetch(f.url, f.max_chars)
        except Blocked as e:
            raise HTTPException(403, f"blocked: {e}") from e
        except HTTPException:
            raise
        except Exception as e:  # navigation errors, timeouts
            msg = str(e).splitlines()[0][:300]
            status = 403 if "ERR_TUNNEL_CONNECTION_FAILED" in msg or "403" in msg else 502
            raise HTTPException(status, f"fetch failed: {msg}") from e

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="AIChain headless browser service")
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--uds", default=None, help="serve the API on this Unix socket instead of a port")
    ap.add_argument("--egress-uds", default=None,
                    help="egress proxy socket (production: browser in a private network namespace)")
    ap.add_argument("--skip-ddg", action="store_true",
                    help="go straight to the Wikipedia fallback (e.g. when DuckDuckGo challenges this IP)")
    ap.add_argument("--no-chromium-sandbox", action="store_true",
                    help="only if the kernel forbids Chromium's sandbox; keep the namespace isolation")
    ap.add_argument("--min-interval", type=float, default=5.0, help="seconds between navigations")
    ap.add_argument("--block", action="append", default=[],
                    help="extra blocked address/network (e.g. this server's public IP)")
    args = ap.parse_args(argv)
    if os.geteuid() == 0:
        print("refusing to run the browser as root; use the aichain-browser user", file=sys.stderr)
        return 2
    extra = args.block + [b for b in os.environ.get("AICHAIN_BLOCK_IPS", "").split(",") if b]
    browser = Browser(EgressGuard(extra_blocked=extra), min_interval=args.min_interval,
                      egress_uds=args.egress_uds, chromium_sandbox=not args.no_chromium_sandbox,
                      skip_ddg=args.skip_ddg)
    if args.uds:
        uvicorn.run(create_app(browser), uds=args.uds, log_level="info")
    else:
        uvicorn.run(create_app(browser), host="127.0.0.1", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
