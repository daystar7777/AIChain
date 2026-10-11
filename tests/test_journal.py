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
