"""M7 analyses: pair payments and collusion flags, answer competition, research vs acceptance,
persona convergence, and the report section built from a replayed chain."""

import time

from aichain import analysis
from aichain.board import Board
from aichain.chain.store import ChainStore
from aichain.gateways.web import FakeWebProvider, SearchGateway
from aichain.journal.store import Journal
from aichain.gateways.llm.base import LLMRequest, LLMResponse
from aichain.memory import KnowledgeStore
from aichain.sim.brains import event_of, persona_brain, persona_of, tool_response
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS, build_network

PAGE = "https://example.org/reverse"


def nomad_searches_first(request: LLMRequest) -> LLMResponse:
    """persona_brain, except that n4 searches the web before answering (and is never chosen,
    because its answer is less confident than n2's)."""
    kind, attrs, _ = event_of(request)
    if kind == "bounty_new" and persona_of(request) == "nomad":
        done = sum(m.role == "tool" for m in request.messages)
        if done == 0:
            return tool_response(("search", {"query": "python list reverse in place", "k": 3}))
        if done == 1:
            return tool_response(("answer", {"bounty_id": attrs["bounty_id"], "answer": {
                "answer": "list.reverse()를 쓴다.", "confidence": 0.5, "sources": [PAGE],
                "verification": "x=[1,2]; x.reverse(); assert x == [2,1]"}}))
        return tool_response(("done", {"summary": "답했다."}))
    return persona_brain(request)


def run(tmp_path, questions=3):
    cdb, jdb, kdb = tmp_path / "chain" / "chain.db", tmp_path / "j.db", tmp_path / "k.db"
    web = FakeWebProvider(results={"python": [{"title": "t", "url": PAGE, "snippet": "s"}]},
                          pages={PAGE: {"title": "t", "text": "list.reverse() reverses in place"}})
    net = build_network("scripted", brain=nomad_searches_first, web=SearchGateway(web),
                        knowledge=KnowledgeStore(kdb), chain=ChainStore(cdb))
    journal = Journal(jdb)
    journal.start_run("t", {})
    journal.attach(net)
    p = SHORT_PARAMS
    for _ in range(questions):
        net.bridge_question("n1", "t", **DEFAULT_QUESTION)
        net.run_until_idle()
        net.advance(p.commit_blocks + p.reveal_blocks + p.judge_blocks + 2)
        net.run_until_idle()
    journal.close()
    return net, cdb, jdb, kdb


def test_analysis_from_a_replayed_chain(tmp_path):
    net, cdb, jdb, kdb = run(tmp_path)
    board, clock = analysis.load_chain(cdb)
    assert board.state_root() == net.board.state_root()
    a = analysis.analyse(board, clock, jdb, kdb, None)
    c = a["competition"]
    assert c["bounties"] == 3 and c["answerers_per_bounty"] == {"2": 3}
    assert c["asked"]["n1"] == 3 and c["answers"] == {"n1": 0, "n2": 3, "n3": 0, "n4": 3}
    assert c["wins"]["n2"] == 3
    n1n2 = next(r for r in a["pairs"] if r["pair"] == "n1-n2")
    assert n1n2["a_to_b_mP"] == 3 * 20_000 and n1n2["b_to_a_mP"] == 0
    # n1 is the only node that asks, so all of n2's income coming from n1 is expected, not captive.
    assert n1n2["b_income_share_from_a"] == 1.0 and not any("수입의" in f for f in n1n2["flags"])
    assert n1n2["accept_n1_of_n2"] == (3, 1.0, None)
    # n4 searched before every answer and lost each time; n2 answered straight away and won.
    assert a["research"]["조사 후"] == {"answers": 3, "accepted": 0, "shared": 0, "rate": 0.0}
    assert a["research"]["바로"]["accepted"] == 3
    assert a["persona"]["mean_start"] == a["persona"]["mean_end"]  # nobody edited a persona
    md = analysis.render_markdown(a)
    assert "## M7 분석" in md and "| n4 | 0 | 3 | 0 |" in md and "| 조사 후 | 3 | 0 | 0% | 0 |" in md


def test_report_includes_the_analysis_when_the_chain_is_there(tmp_path):
    from aichain import report

    _, cdb, jdb, kdb = run(tmp_path, questions=1)
    out = tmp_path / "r"
    assert report.main(["--all", "--journal", str(jdb), "--knowledge", str(kdb), "--out", str(out),
                        "--chain", str(cdb)]) == 0
    md = next(out.glob("*.md")).read_text()
    assert "## M7 분석" in md and "### 담합 신호 (노드 쌍)" in md and "### 성향 수렴" in md
    assert md.index("## M7 분석") < md.index("## 노드별 활동")
    out2 = tmp_path / "r2"
    assert report.main(["--all", "--journal", str(jdb), "--knowledge", str(kdb), "--out", str(out2),
                        "--chain", str(tmp_path / "missing.db")]) == 0
    assert "## M7 분석" not in next(out2.glob("*.md")).read_text()


def test_pair_flags():
    board = Board(SHORT_PARAMS)
    fl = [{"height": 1, "payer": "n1", "payee": "n2", "kind": "transfer_in", "mP": 30_000},
          {"height": 2, "payer": "n2", "payee": "n1", "kind": "transfer_in", "mP": 25_000},
          {"height": 3, "payer": "n3", "payee": "n4", "kind": "info_sold", "mP": 5_000},
          {"height": 4, "payer": None, "payee": "n4", "kind": "info_sold", "mP": 5_000}]
    rows = {r["pair"]: r for r in analysis.pairs(board, fl)}
    assert rows["n1-n2"]["reciprocity"] == 0.83 and rows["n1-n2"]["flags"] == ["주고받기"]
    assert rows["n3-n4"]["b_income_share_from_a"] == 1.0
    assert rows["n3-n4"]["flags"] == []  # below the 10P volume floor
    assert rows["n1-n3"]["trades"] == 0 and rows["n1-n3"]["flags"] == []
    # n3 spends a fifth of what the nodes spend but pays all of n4's income: captive.
    fl[2]["mP"] = 15_000
    rows = {r["pair"]: r for r in analysis.pairs(board, fl)}
    assert rows["n3-n4"]["flags"] == ["n4 수입의 100%가 n3에게서"]
    # The same 15P from a node that also pays most of everyone else is just its weight.
    fl2 = [{"height": 1, "payer": "n1", "payee": p, "kind": "answer_accepted", "mP": 15_000}
           for p in ("n2", "n3", "n4")]
    assert all(r["flags"] == [] for r in analysis.pairs(board, fl2))


def test_persona_convergence_and_clock():
    base = {"n1": "질문을 많이 하고 답을 비교한다. 출처를 중시한다.",
            "n2": "코드를 짜고 테스트로 검증한다. 간결하게 답한다."}
    later = base["n2"] + " 출처를 중시한다."
    rows = {"n1": [{"ts": 100.0, "data": {"verdict": "approved", "text": later}}],
            "n2": [{"ts": 50.0, "data": {"verdict": "rejected", "text": "무시됨"}}]}
    tl = analysis.persona_timeline(base, rows)
    assert len(tl["n1"]) == 2 and len(tl["n2"]) == 1  # a rejected edit changes nothing
    c = analysis.convergence(tl, 10.0, 200.0, [20.0, 150.0])
    assert c["mean_end"] > c["mean_start"] and c["edits"] == {"n1": 1, "n2": 0}
    assert c["drift"]["n2"] == 0.0 and c["drift"]["n1"] > 0
    assert [round(v, 4) for _, v in c["series"]] == [c["mean_start"], c["mean_end"]]
    assert analysis.similarity("같은 문장", "같은 문장") == 1.0 and analysis.similarity("", "x") == 0.0
    now = time.time()
    clock = analysis.Clock([(0, now - 7200), (60, now - 3600), (120, now)])
    assert clock.ts(90) == now - 3600 and clock.height_at(now - 3600) == 60
    assert analysis.Clock([]).height_at(now) == 0 and analysis.Clock([]).ts(5) is None


def test_daily_issuance_counts_the_first_day(tmp_path):
    import json
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metrics (run_id INTEGER, ts REAL, height INTEGER, data TEXT)")
    day = 86_400.0
    t0 = time.mktime(time.strptime("2026-10-10 12:00", "%Y-%m-%d %H:%M"))
    for ts, h, issued, burned in ((t0 - day, 10, 100, 0), (t0, 20, 150, 5), (t0 + 3600, 30, 180, 9),
                                  (t0 + day, 40, 260, 9)):
        m = {"height": h, "supply": {"circulating_mP": issued - burned, "issued_mP": issued, "burned_mP": burned},
             "balances_mP": {"n1": issued - burned}}
        conn.execute("INSERT INTO metrics VALUES (1, ?, ?, ?)", (ts, h, json.dumps(m)))
    rows = analysis.metrics_by_day(conn, t0 - 60)
    # The first day in range is measured from the last row before it, not left blank.
    assert [(r["day"], r["issued_day_mP"], r["burned_day_mP"]) for r in rows] == [
        ("2026-10-10", 80, 9), ("2026-10-11", 80, 0)]
    rows = analysis.metrics_by_day(conn, None)
    assert rows[0]["issued_day_mP"] == 0 and rows[1]["issued_day_mP"] == 80
