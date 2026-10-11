"""Parse DuckDuckGo's HTML results page (html.duckduckgo.com/html)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urlsplit

from bs4 import BeautifulSoup

SEARCH_URL = "https://html.duckduckgo.com/html/"


class SearchBlocked(Exception):
    """The engine answered with a bot check instead of results. Never bypassed."""


def search_url(query: str, region: str | None = None) -> str:
    params = {"q": query}
    if region:
        params["kl"] = region
    return SEARCH_URL + "?" + urlencode(params)


def _unwrap(href: str) -> str | None:
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        target = parse_qs(parts.query).get("uddg", [None])[0]
        return target
    if parts.scheme in ("http", "https"):
        return href
    return None


def parse_results(html: str, k: int = 8) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True).lower()
    if soup.select_one("form#challenge-form, .anomaly-modal") or re.search(
            r"bots use duckduckgo too|unusual traffic|complete the following challenge", text):
        raise SearchBlocked("DuckDuckGo returned a bot check")
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in soup.select("div.result"):
        classes = block.get("class") or []
        if "result--ad" in classes:
            continue
        link = block.select_one("a.result__a")
        if link is None or not link.get("href"):
            continue
        url = _unwrap(link["href"])
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = block.select_one(".result__snippet")
        results.append({
            "title": link.get_text(" ", strip=True),
            "url": url,
            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
        })
        if len(results) >= k:
            break
    return results


# ---- Wikipedia (official MediaWiki search API; allowed for automated clients) ----

def wikipedia_search_url(query: str, k: int) -> tuple[str, str]:
    lang = "ko" if re.search(r"[가-힣]", query) else "en"
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": str(k),
              "format": "json", "utf8": "1"}
    return lang, f"https://{lang}.wikipedia.org/w/api.php?" + urlencode(params)


def parse_wikipedia(lang: str, payload: dict, k: int) -> list[dict[str, str]]:
    out = []
    for hit in (payload.get("query") or {}).get("search", [])[:k]:
        title = hit.get("title", "")
        snippet = BeautifulSoup(hit.get("snippet", ""), "html.parser").get_text(" ", strip=True)
        out.append({"title": title, "snippet": snippet,
                    "url": f"https://{lang}.wikipedia.org/wiki/" + title.replace(" ", "_")})
    return out
