import json
import sqlite3

from fastapi.testclient import TestClient

from aichain.gateways.llm.base import ProviderError
from aichain.journal.store import REDACTED, Journal, redact
from aichain.journal.viewer import create_app
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS, build_network

FAKE_KEY = "sk-" + "A1b2C3d4" * 6
FAKE_TOKEN = "acb_" + "x9Y8z7W6" * 5


def run_demo(db, brain=None, question=DEFAULT_QUESTION):
    kw = {"brain": brain} if brain else {}
    net = build_network("scripted", **kw)
    journal = Journal(db)
    run = journal.start_run("test", {"api_key": FAKE_KEY, "provider": "scripted"})
    journal.attach(net)
    net.bridge_question("n1", "test", **question)
    net.run_until_idle()
    p = SHORT_PARAMS
    net.advance(p.commit_blocks + p.reveal_blocks + p.judge_blocks + p.dispute_blocks + 2)
    journal.close()
    return run, net


def all_text(db):
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT data FROM entries UNION ALL SELECT meta FROM runs "
                        "UNION ALL SELECT data FROM snapshots").fetchall()
    return "\n".join(r[0] for r in rows)


def test_redact_masks_keys_tokens_and_headers():
    data = {
        "Authorization": "Bearer abc.def.ghi",
        "headers": {"authorization": "x", "x-api-key": "y"},
        "api_key": FAKE_KEY,
        "note": f"key={FAKE_KEY} token {FAKE_TOKEN} hdr: Bearer {FAKE_KEY}",
        "env": "OPENCODE_GO_API_KEY=whatever123",
        "usage": {"input_tokens": 10, "output_tokens": 3},
        "list": [FAKE_KEY],
    }
    out = redact(data)
    text = json.dumps(out)
    assert FAKE_KEY not in text and FAKE_TOKEN not in text and "abc.def.ghi" not in text
    assert "whatever123" not in text
    assert out["Authorization"] == REDACTED and out["headers"]["x-api-key"] == REDACTED
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert out["env"] == "OPENCODE_GO_API_KEY=" + REDACTED


def test_journal_records_everything_and_never_secrets(tmp_path):
    db = tmp_path / "j.db"
    leaky = {**DEFAULT_QUESTION, "body": f"내 키는 {FAKE_KEY} 이고 토큰은 Bearer {FAKE_TOKEN} 이다"}
    run, net = run_demo(db, question=leaky)
    conn = sqlite3.connect(db)
    kinds = dict(conn.execute("SELECT kind, COUNT(*) FROM entries WHERE run_id=? GROUP BY kind",
                              (run,)).fetchall())
    for k in ("episode_start", "episode_end", "llm_call", "tool", "reveal_open", "judge_due",
              "bounty_settled", "bounty_final", "balance", "memory", "duty_reveal"):
        assert kinds.get(k), k
    # Full prompt and response are stored for each LLM call.
    data = json.loads(conn.execute("SELECT data FROM entries WHERE kind='llm_call' LIMIT 1").fetchone()[0])
    assert data["messages"][0]["role"] == "system" and "마스터마인드" in data["messages"][0]["content"]
    assert data["response"]["tool_calls"][0]["name"] == "ask"
    assert "latency_ms" in data and data["response"]["usage"]["input_tokens"] > 0
    text = all_text(db)
    assert FAKE_KEY not in text and FAKE_TOKEN not in text
    assert REDACTED in text


def test_failed_llm_call_is_journaled_redacted(tmp_path):
    db = tmp_path / "j.db"

    def broken(request):
        raise ProviderError(f"HTTP 401: invalid key Bearer {FAKE_KEY}", status=401)

    run_demo(db, brain=broken)
    conn = sqlite3.connect(db)
    data = json.loads(conn.execute("SELECT data FROM entries WHERE kind='llm_call'").fetchone()[0])
    assert data["status"] == 401 and REDACTED in data["error"]
    assert FAKE_KEY not in all_text(db)


def test_viewer_api(tmp_path):
    db = tmp_path / "j.db"
    run, _ = run_demo(db)
    client = TestClient(create_app(db, poll_seconds=0.01))
    assert "AIChain 저널" in client.get("/").text
    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == run and runs[0]["ended_at"] and runs[0]["meta"]["api_key"] == REDACTED

    entries = client.get(f"/api/runs/{run}/entries").json()
    assert entries and all(e["data"] is not None for e in entries)
    only = client.get(f"/api/runs/{run}/entries", params={"node": "n2", "kind": "tool"}).json()
    assert only and all(e["node"] == "n2" and e["kind"] == "tool" for e in only)

    state = client.get(f"/api/runs/{run}/state").json()
    assert [n["id"] for n in state["nodes"]] == ["n1", "n2", "n3", "n4"]
    (b,) = state["bounties"]
    assert b["state"] == "FINAL" and any(a["accepted"] for a in b["answers"])
    assert state["supply"]["balanced"] is True

    ep = next(e["episode"] for e in entries if e["kind"] == "episode_start" and e["node"] == "n1")
    steps = client.get(f"/api/runs/{run}/episodes/{ep}").json()
    assert [s["kind"] for s in steps][:2] == ["episode_start", "llm_call"]
    assert steps[-1]["kind"] == "episode_end"
    assert client.get(f"/api/runs/{run}/episodes/nope").status_code == 404


def test_viewer_is_read_only(tmp_path):
    db = tmp_path / "j.db"
    run_demo(db)
    before = db.stat().st_mtime_ns
    client = TestClient(create_app(db))
    client.get("/api/runs")
    client.get("/api/runs/1/entries")
    assert client.post("/api/runs").status_code == 405
    assert db.stat().st_mtime_ns == before


def test_viewer_without_database(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.db"))
    assert client.get("/api/runs").status_code == 503


def test_viewer_knowledge_endpoint(tmp_path):
    from aichain import setup as node_setup
    from aichain.gateways.web import FakeWebProvider, SearchGateway
    from aichain.memory import KnowledgeStore
    from aichain.sim.brains import research_brain

    page = "https://example.org/py"
    web = FakeWebProvider(results={"Python": [{"title": "t", "url": page, "snippet": "s"}]},
                          pages={page: {"title": "Py", "text": "Python fact text"}})
    kdb = tmp_path / "k.db"
    net = build_network("scripted", brain=research_brain, web=SearchGateway(web),
                        knowledge=KnowledgeStore(kdb), interests=node_setup.DEFAULT_INTERESTS)
    net.research(["n2"])
    net.run_until_idle()
    client = TestClient(create_app(tmp_path / "j.db", knowledge_path=kdb))
    k = client.get("/api/knowledge").json()["nodes"]["n2"]
    assert k["notes"][0]["topic"] == "Python 개발" and k["notes"][0]["version"] == 1
    assert k["notes"][0]["facts"][0]["source_url"] == page
    assert k["facts"][0]["url"] == page
    assert client.get("/api/knowledge", params={"node": "n1"}).json()["nodes"] == {}


def test_metrics_series_and_endpoint(tmp_path):
    from aichain.metrics import gini, vote_agreement

    assert gini([1, 1, 1, 1]) == 0.0 and gini([0, 0, 0, 4]) == 0.75
    db = tmp_path / "j.db"
    run, net = run_demo(db)
    client = TestClient(create_app(db))
    rows = client.get(f"/api/runs/{run}/metrics").json()
    assert len(rows) >= 2 and rows[-1]["data"]["bounties"] == {"PAID": 1}
    last = rows[-1]["data"]
    assert last["reputation"]["n2"] > last["reputation"]["n3"]
    assert last["supply"]["circulating_mP"] + last["supply"]["escrow_mP"] + last["supply"]["burned_mP"] == last["supply"]["issued_mP"]

    class P:
        def __init__(self, votes):
            self.votes = votes
    assert vote_agreement([P({"a": "yes", "b": "yes"}), P({"a": "yes", "b": "no"})]) == {"a-b": 0.5}


def test_report_generation(tmp_path):
    from aichain import report
    from aichain import setup as node_setup
    from aichain.gateways.web import FakeWebProvider, SearchGateway
    from aichain.memory import KnowledgeStore
    from aichain.sim.brains import research_brain

    page = "https://example.org/py"
    web = FakeWebProvider(results={"Python": [{"title": "t", "url": page, "snippet": "s"}]},
                          pages={page: {"title": "Py", "text": "Python fact text"}})
    jdb, kdb = tmp_path / "j.db", tmp_path / "k.db"
    net = build_network("scripted", brain=research_brain, web=SearchGateway(web),
                        knowledge=KnowledgeStore(kdb), interests=node_setup.DEFAULT_INTERESTS)
    journal = Journal(jdb)
    journal.start_run("t", {})
    journal.attach(net)
    net.research(["n2"])
    net.run_until_idle()
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    net.advance(SHORT_PARAMS.commit_blocks + SHORT_PARAMS.reveal_blocks + SHORT_PARAMS.judge_blocks + 2)
    journal.close()
    assert report.main(["--all", "--journal", str(jdb), "--knowledge", str(kdb), "--out", str(tmp_path / "r")]) == 0
    md = next((tmp_path / "r").glob("*.md")).read_text()
    assert "## 지식 체계" in md and "Python 개발" in md and "| n2 |" in md
    assert "example.org 1" in md
    html_text = next((tmp_path / "r").glob("*.html")).read_text()
    assert "<table>" in html_text and "<h2>경제와 평판</h2>" in html_text


def test_viewer_serves_reports(tmp_path):
    rdir = tmp_path / "reports"
    rdir.mkdir()
    (rdir / "report-20261012-0010-24h.html").write_text("<h1>r</h1>")
    client = TestClient(create_app(tmp_path / "j.db", reports_dir=rdir))
    assert "report-20261012-0010-24h" in client.get("/reports").text
    assert client.get("/reports/report-20261012-0010-24h.html").text == "<h1>r</h1>"
    for bad in ("../j.db", "x.html", "report-x.sh"):
        assert client.get(f"/reports/{bad}").status_code == 404


def test_snapshot_lists_open_and_recent_items_only():
    from aichain.journal.store import network_snapshot

    net = build_network("scripted")
    net.bridge_question("n1", "test", **DEFAULT_QUESTION)
    net.run_until_idle()
    snap = network_snapshot(net, recent=0)
    assert [b["state"] for b in snap["bounties"]] == ["OPEN"]  # open ones always
    assert snap["totals"]["bounties"] == 1 and "metrics" not in snap
    p = SHORT_PARAMS
    net.advance(p.commit_blocks + p.reveal_blocks + p.judge_blocks + p.dispute_blocks + 2)
    snap = network_snapshot(net, recent=0)
    assert snap["bounties"] == [] and snap["totals"]["bounties"] == 1  # closed: in the journal only
    assert [b["state"] for b in network_snapshot(net, recent=1)["bounties"]] == ["FINAL"]


def test_snapshot_is_written_by_the_writer_thread_at_most_every_interval(tmp_path):
    import threading
    import time

    net = build_network("scripted")
    journal = Journal(tmp_path / "j.db", snapshot_interval=3600)
    journal.start_run("t")
    writes, captures = [], []
    real_write, real_capture = journal._write, journal._capture
    journal._write = lambda *a: (writes.append(threading.current_thread().name), real_write(*a))
    journal._capture = lambda force: (captures.append(force), real_capture(force))[1]
    journal.attach(net)  # first snapshot inline
    net.bridge_question("n1", "test", **DEFAULT_QUESTION)
    net.run_until_idle()  # dozens of log entries within a fraction of the interval
    assert captures == [True]
    journal._last_snapshot -= 3600  # the interval has passed
    net.advance(1)
    journal.tick()
    deadline = time.time() + 5
    while len(writes) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert captures == [True, False] and writes[1] == "journal-writer"
    journal._last_snapshot -= 3600
    journal.tick()  # due, but nothing changed: no capture
    assert len(captures) == 2
    journal.close()
    assert not any(t.name == "journal-writer" for t in threading.enumerate())
    state = TestClient(create_app(tmp_path / "j.db")).get(f"/api/runs/{journal.run_id}/state").json()
    assert state["height"] == net.board.height and state["bounties"][0]["state"] == "OPEN"


def test_snapshot_writer_survives_a_full_disk(tmp_path, capsys):
    import time

    net = build_network("scripted")
    journal = Journal(tmp_path / "j.db", snapshot_interval=0)
    journal.start_run("t")
    journal.attach(net)
    real_write, calls = journal._write, []

    def flaky(*a):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database or disk is full")
        real_write(*a)

    journal._write = flaky
    for _ in range(2):
        net.advance(1)
        journal.tick()
        deadline = time.time() + 5
        while journal._pending is not None and time.time() < deadline:
            time.sleep(0.01)
    time.sleep(0.05)
    journal.close()
    assert journal.write_errors == 1 and len(calls) >= 2
    assert "disk is full" in capsys.readouterr().err


def test_prune_works_in_chunks_and_stops_at_young_entries(tmp_path):
    import time

    j = Journal(tmp_path / "j.db", keep_prompts_days=1)
    j.PRUNE_CHUNK = 2
    j.start_run("t")
    for i in range(5):
        j.record("llm_call", {"messages": [{"role": "user", "content": f"p{i}"}], "response": {"text": "ok"}})
    j.record("tool", {"tool": "x"})
    assert j.prune(now=time.time() + 2 * 86400) == 5
    assert j.prune(now=time.time() + 2 * 86400) == 0  # resumes after the last pruned id
    j.record("llm_call", {"messages": [{"role": "user", "content": "new"}]})
    assert j.prune() == 0 and j._pruned_upto == 6  # the new entry is too young
    rows = [json.loads(d) for (d,) in sqlite3.connect(tmp_path / "j.db").execute(
        "SELECT data FROM entries WHERE kind = 'llm_call' ORDER BY id")]
    assert all(r.get("messages_pruned") for r in rows[:5]) and rows[5]["messages"][0]["content"] == "new"


def test_viewer_finds_older_items_by_id(tmp_path):
    db = tmp_path / "j.db"
    run, net = run_demo(db)
    (bid,) = net.board.bounties
    client = TestClient(create_app(db))
    rows = client.get("/api/find", params={"ref": bid}).json()
    assert rows and all(bid in json.dumps(r["data"], ensure_ascii=False) for r in rows)
    assert {"bounty_settled", "bounty_final"} <= {r["kind"] for r in rows}
    assert all(r["run_id"] == run and r["kind"] != "llm_call" for r in rows)
    assert client.get("/api/find", params={"ref": "x' OR 1=1 --"}).status_code == 422


def test_report_aggregates_match_a_full_read_and_respect_the_period(tmp_path):
    import time
    from collections import Counter

    from aichain import report

    db = tmp_path / "j.db"
    run_demo(db)
    conn = sqlite3.connect(db)
    old = time.time() - 3 * 86400  # half of the entries are from three days ago
    conn.execute("UPDATE entries SET ts = ? WHERE id <= (SELECT MAX(id) / 2 FROM entries)", (old,))
    conn.execute("UPDATE metrics SET ts = ? WHERE rowid = 1", (old,))
    conn.commit()
    j = Journal(db)  # an extra entry: a web fetch, recent
    j.start_run("t")
    j.record("web_fetch", {"url": "https://example.org/a"})
    j.record("llm_call", {"error": "HTTP 500"})
    j.close()

    def reference(since):  # what the old report computed from every row in memory
        rows = [(k, n, json.loads(d)) for k, n, d in conn.execute(
            "SELECT kind, node, data FROM entries WHERE ts >= ? ORDER BY id", (since,))]
        llm = [d for k, _, d in rows if k == "llm_call"]
        return (Counter(k for k, _, _ in rows),
                sum((d.get("response") or {}).get("usage", {}).get("input_tokens", 0) for d in llm),
                sum((d.get("response") or {}).get("usage", {}).get("output_tokens", 0) for d in llm),
                sum(1 for d in llm if d.get("error")),
                Counter((n, d.get("tool")) for k, n, d in rows if k == "tool"))

    for since in (None, time.time() - 86400):
        a = report.analyse(report.collect(db, tmp_path / "none.db", since))
        by_kind, tin, tout, failed, tools = reference(since or 0)
        assert a["by_kind"] == by_kind and a["tokens"] == (tin, tout) and tin > 0
        assert a["failed_calls"] == failed == 1 and a["llm_calls"] == by_kind["llm_call"]
        assert Counter({(n, t): c for n, ts in a["tools_used"].items() for t, c in ts.items()}) == tools
        assert a["domains"] == Counter({"example.org": 1})
    day = report.collect(db, tmp_path / "none.db", time.time() - 86400)
    assert sum(day["by_kind"].values()) < sum(report.collect(db, tmp_path / "none.db", None)["by_kind"].values())
    assert day["first"]["height"] > 0 and day["period"][0] >= time.time() - 86400
    # The day's rows are found through the index, not by reading the whole table.
    plan = " ".join(r[3] for r in conn.execute(
        "EXPLAIN QUERY PLAN SELECT kind, COUNT(*) FROM entries WHERE ts >= ? GROUP BY kind", (0,)))
    assert "entries_ts" in plan


def test_timeline_api_leaves_out_llm_prompts(tmp_path):
    db = tmp_path / "j.db"
    run, _ = run_demo(db)
    client = TestClient(create_app(db))
    calls = client.get(f"/api/runs/{run}/entries", params={"kind": "llm_call"}).json()
    assert calls and all("messages" not in c["data"] and c["data"]["response"] for c in calls)
    ep = calls[0]["episode"]
    steps = client.get(f"/api/runs/{run}/episodes/{ep}").json()
    assert next(s for s in steps if s["kind"] == "llm_call")["data"]["messages"]  # the episode view keeps them
