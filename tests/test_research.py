import asyncio
import json
from pathlib import Path

import pytest

from aichain import setup as node_setup
from aichain.browser.ddg import SearchBlocked, parse_results, search_url
from aichain.browser.guard import Blocked, EgressGuard
from aichain.browser.proxy import GuardProxy
from aichain.gateways.llm.base import (
    CallLimitReached,
    GatewayConfig,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    Message,
    TokenBucket,
    Usage,
)
from aichain.gateways.llm.scripted import tool_response
from aichain.gateways.web import FakeWebProvider, SearchGateway, WebError, WebLimits
from aichain.memory import KnowledgeStore, Memory
from aichain.sim.brains import research_brain
from aichain.sim.factory import build_network

FIXTURES = Path(__file__).parent / "fixtures"


def run(coro):
    return asyncio.run(coro)


def fixed_resolver(table):
    async def resolve(host, port):
        if host not in table:
            raise OSError("NXDOMAIN")
        return table[host]
    return resolve


# ---- SSRF guard ----

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:80/", "http://10.1.2.3/", "http://192.168.0.10/",
    "http://172.20.0.1/", "http://169.254.169.254/latest/meta-data/", "http://100.98.150.35:80/",
    "http://100.99.152.31/", "http://[::1]/", "http://[fc00::1]/", "http://[::ffff:127.0.0.1]/",
    "http://0.0.0.0/", "http://46.250.255.147/", "file:///etc/passwd", "ftp://example.com/",
    "http://example.com:22/", "http://user:pw@example.com/", "http://printer.local/",
])
def test_guard_blocks_internal_targets(url):
    guard = EgressGuard(extra_blocked=["46.250.255.147"],
                        resolver=fixed_resolver({"example.com": ["93.184.215.14"]}))
    with pytest.raises(Blocked):
        run(guard.check_url(url))


def test_guard_allows_public_and_checks_every_answer():
    guard = EgressGuard(resolver=fixed_resolver({
        "example.com": ["93.184.215.14"],
        "rebind.example": ["93.184.215.14", "10.0.0.5"],  # one private answer is enough to block
        "tailnet.example": ["100.98.150.35"],
    }))
    assert run(guard.check_url("https://example.com/a?b=1")) == ["93.184.215.14"]
    for host in ("rebind.example", "tailnet.example"):
        with pytest.raises(Blocked):
            run(guard.check_url(f"https://{host}/"))
    with pytest.raises(Blocked):
        run(guard.check_url("https://nxdomain.example/"))


async def _http(port, raw: bytes) -> bytes:
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(raw)
    await w.drain()
    data = await r.read()
    w.close()
    return data


def test_proxy_blocks_and_forwards_to_the_checked_address():
    async def scenario():
        # Origin server standing in for a public site.
        async def origin(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(origin, "127.0.0.1", 0)
        origin_port = server.sockets[0].getsockname()[1]
        answers = iter([["127.0.0.1"], ["10.9.9.9"]])  # DNS changes after the first lookup

        async def rebinding(host, port):
            return next(answers)

        blocked = []
        guard = EgressGuard(allowed_ports={80, 443, origin_port}, resolver=rebinding,
                            allow_hosts={"site.test"})
        proxy = GuardProxy(guard, on_block=lambda t, r: blocked.append(r))
        port = await proxy.start()
        ok = await _http(port, f"GET http://site.test:{origin_port}/x HTTP/1.1\r\nHost: site.test\r\n\r\n".encode())
        assert ok.endswith(b"hello")
        # Literal internal targets and disallowed ports are refused.
        for raw in (b"GET http://127.0.0.1/ HTTP/1.1\r\nHost: x\r\n\r\n",
                    b"CONNECT 10.0.0.1:443 HTTP/1.1\r\nHost: x\r\n\r\n",
                    b"CONNECT 169.254.169.254:443 HTTP/1.1\r\n\r\n",
                    b"CONNECT example.com:22 HTTP/1.1\r\n\r\n"):
            resp = await _http(port, raw)
            assert resp.startswith(b"HTTP/1.1 403"), resp
        assert len(blocked) == 4
        await proxy.close()
        server.close()

    run(scenario())


# ---- DuckDuckGo parser ----

def test_parse_ddg_results():
    results = parse_results((FIXTURES / "ddg_results.html").read_text(), k=8)
    assert [r["url"] for r in results] == [
        "https://docs.python.org/3.14/whatsnew/3.14.html", "https://peps.python.org/pep-0745/"]
    assert results[0]["title"] == "What's New In Python 3.14"
    assert "new features" in results[0]["snippet"]
    assert len(parse_results((FIXTURES / "ddg_results.html").read_text(), k=1)) == 1
    assert search_url("a b").startswith("https://html.duckduckgo.com/html/?q=a+b")


def test_parse_ddg_challenge_is_reported_not_bypassed():
    with pytest.raises(SearchBlocked):
        parse_results((FIXTURES / "ddg_challenge.html").read_text())


# ---- search gateway ----

def test_search_gateway_limits_and_per_node_cache():
    now = [0.0]
    web = FakeWebProvider(results={"py": [{"title": "t", "url": "https://a.example/", "snippet": "s"}]})
    gw = SearchGateway(web, WebLimits(search_per_hour=2, fetch_per_hour=1), clock=lambda: now[0])
    gw.search("n1", "py news", 3)
    assert gw.search("n1", "py news", 3)["cached"] is True
    gw.search("n2", "py news", 3)  # other node: not shared
    assert len(web.calls) == 2
    gw.search("n1", "py other", 3)
    with pytest.raises(WebError):
        gw.search("n1", "py third", 3)
    now[0] = 3601
    gw.search("n1", "py fourth", 3)


# ---- research episodes ----

PAGE = "https://docs.python.org/3.14/whatsnew/3.14.html"


def research_net(tmp_path, brain=research_brain, **kw):
    web = FakeWebProvider(
        results={"Python": [{"title": "What's new", "url": PAGE, "snippet": "3.14 features"}]},
        pages={PAGE: {"title": "What's New In Python 3.14",
                      "text": "Python 3.14 adds template strings (PEP 750) and deferred annotations."}})
    store = KnowledgeStore(tmp_path / "k.db")
    net = build_network("scripted", brain=brain, web=SearchGateway(web), knowledge=store,
                        interests=node_setup.DEFAULT_INTERESTS, **kw)
    return net, web, store


def test_research_episode_builds_a_note_with_sources(tmp_path):
    net, web, store = research_net(tmp_path)
    net.research(["n2"])
    net.run_until_idle()
    n2 = net.nodes["n2"]
    (note,) = n2.memory.notes()
    assert note.topic == "Python 개발" and note.trust == 0.3
    assert note.data["facts"][0]["source_url"] == PAGE
    fact = next(r for r in n2.memory.records if r.kind == "fact")
    assert fact.source == f"web:{PAGE}" and fact.trust == 0.3
    assert n2.memory.verify_chain()
    kinds = [e["event"] for e in net.events]
    assert "web_search" in kinds and "web_fetch" in kinds and "knowledge" in kinds
    assert web.calls == [("search", "Python 개발 최신"), ("fetch", PAGE)]
    # The research prompt carried the interests and the fixed instructions.
    assert [e.event for e in net.episodes] == ["research_tick"]

    # Knowledge survives a restart, and a second pass supersedes the note.
    store2 = KnowledgeStore(tmp_path / "k.db")
    reloaded = Memory("n2", store2)
    assert reloaded.verify_chain() and reloaded.latest_note("Python 개발") is not None
    net2 = build_network("scripted", brain=research_brain, web=net.web, knowledge=store2,
                         interests=node_setup.DEFAULT_INTERESTS)
    net2.research(["n2"])
    net2.run_until_idle()
    m = net2.nodes["n2"].memory
    assert len(m.notes()) == 1 and m.notes()[0].supersedes == note.hash
    assert "Python 개발" in net2.nodes["n2"]._notes_index()


def test_web_sources_must_be_pages_the_node_fetched(tmp_path):
    def liar(request):
        if request.messages[-1].role == "tool":
            return tool_response(("done", {"summary": "x"}))
        return tool_response(
            ("note", {"topic": "AI", "summary": "s", "confidence": 0.9,
                      "facts": [{"text": "made up", "source_url": "https://never.fetched/"}]}),
            ("remember", {"kind": "fact", "content": "c", "source_url": "https://never.fetched/"}))

    net, _, _ = research_net(tmp_path, brain=liar)
    net.research(["n1"])
    net.run_until_idle()
    assert not net.nodes["n1"].memory.notes()
    assert sum(1 for e in net.events if e["event"] == "board_reject") == 2


def test_web_tools_hidden_without_gateway_or_when_conserving(tmp_path):
    net, _, _ = research_net(tmp_path)
    n1 = net.nodes["n1"]
    assert {"search", "fetch", "note"} <= set(n1.tool_names())
    n1.bucket.charge(n1.bucket.capacity * 0.8)
    assert not {"search", "fetch", "note"} & set(n1.tool_names())
    plain = build_network("scripted")
    assert "search" not in plain.nodes["n1"].tool_names()


def test_answer_must_cite_sources_after_using_the_web(tmp_path):
    net, _, _ = research_net(tmp_path)
    node = net.nodes["n2"]
    node._used_web = True  # as if search/fetch ran earlier in this episode
    args = json.dumps({"bounty_id": "b_x", "answer": {"answer": "a", "confidence": 0.5, "sources": []}})
    assert "sources가 비어" in node._exec_tool("answer", args)
    # Second time the Core lets the protocol decide (here: unknown bounty).
    assert "unknown bounty" in node._exec_tool("answer", args)


def test_research_interval_ticks_idle_nodes(tmp_path):
    net, _, _ = research_net(tmp_path, research_interval_blocks=10)
    net.advance(10)
    assert {e.node_id for e in net.episodes if e.event == "research_tick"} == {"n1", "n2", "n3", "n4"}


def test_daily_call_caps():
    class One:
        name = "x"

        def complete(self, request):
            return LLMResponse("ok", [], Usage(1, 1))

    gw = LLMGateway({"x": One()}, GatewayConfig(daily_calls_per_node=2, daily_calls_total=3))
    for n in ("a", "b"):
        gw.register(n, TokenBucket(10**6, lambda: 0))
    req = lambda n: LLMRequest(n, "m", [Message("user", "hi")], [])
    gw.chat("x", req("a"), 3)
    gw.chat("x", req("a"), 3)
    with pytest.raises(CallLimitReached):
        gw.chat("x", req("a"), 3)
    gw.chat("x", req("b"), 3)
    with pytest.raises(CallLimitReached):
        gw.chat("x", req("b"), 3)


def test_setup_cli_roundtrip(tmp_path):
    f = tmp_path / "nodes.json"
    assert node_setup.load(f)["n2"][0]["topic"] == "Python 개발"  # defaults when missing
    node_setup.main(["--file", str(f), "--defaults"])
    node_setup.main(["--file", str(f), "--node", "n2", "--interest", "Rust: 시스템 언어", "--interest", "WASM"])
    loaded = node_setup.load(f)
    assert loaded["n2"] == [{"topic": "Rust", "description": "시스템 언어"}, {"topic": "WASM", "description": ""}]
    assert loaded["n1"] == node_setup.DEFAULT_INTERESTS["n1"]
    net = build_network("scripted", interests=loaded)
    assert "## 관심사 (초기 셋업에서 정함)\n- Rust: 시스템 언어\n- WASM" in net.nodes["n2"].persona_layer


def test_precheck_without_dns():
    guard = EgressGuard()
    guard.precheck("https://example.com/x")
    for url in ("http://10.0.0.1/", "http://localhost/", "http://[::1]/", "gopher://x/", "http://a.com:8080/"):
        with pytest.raises(Blocked):
            guard.precheck(url)


def test_unix_egress_chain():
    """forwarder (browser namespace) → egress Unix socket → guard → origin."""
    import tempfile

    from aichain.browser.proxy import UnixForwarder

    short_dir = tempfile.mkdtemp(dir="/tmp")  # AF_UNIX paths must be short (macOS: 104 bytes)

    async def scenario():
        async def origin(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(origin, "127.0.0.1", 0)
        oport = server.sockets[0].getsockname()[1]
        guard = EgressGuard(allowed_ports={80, 443, oport}, allow_hosts={"site.test"},
                            resolver=fixed_resolver({"site.test": ["127.0.0.1"]}))
        sock = f"{short_dir}/e.sock"
        egress = GuardProxy(guard)
        await egress.start_unix(sock)
        fwd = UnixForwarder(sock)
        port = await fwd.start()
        ok = await _http(port, f"GET http://site.test:{oport}/ HTTP/1.1\r\nHost: site.test\r\n\r\n".encode())
        assert ok.endswith(b"ok")
        denied = await _http(port, b"CONNECT 100.98.150.35:443 HTTP/1.1\r\n\r\n")
        assert denied.startswith(b"HTTP/1.1 403")
        await fwd.close()
        await egress.close()
        server.close()

    run(scenario())


def test_parse_wikipedia_fallback():
    from aichain.browser.ddg import parse_wikipedia, wikipedia_search_url

    lang, url = wikipedia_search_url("대형 언어 모델", 3)
    assert lang == "ko" and url.startswith("https://ko.wikipedia.org/w/api.php?")
    assert wikipedia_search_url("large language model", 3)[0] == "en"
    payload = {"query": {"search": [{"title": "Large language model",
                                     "snippet": 'A <span class="searchmatch">large</span> model'}]}}
    (hit,) = parse_wikipedia("en", payload, 5)
    assert hit == {"title": "Large language model", "snippet": "A large model",
                   "url": "https://en.wikipedia.org/wiki/Large_language_model"}
