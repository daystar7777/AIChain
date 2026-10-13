import json

import pytest

from aichain.board import Board, BoardError, Params
from aichain.node import STORAGE_BASE_KB
from aichain.sim.factory import build_network


def test_lease_moves_quota_for_its_duration():
    net = build_network("scripted")
    seller, buyer = net.nodes["n2"], net.nodes["n1"]
    out = seller._exec_tool("offer_create", json.dumps({"kind": "storage", "quantity": 200, "title": "200KB",
                                                        "summary": "남는 공간", "price_P": 1, "period_hours": 1}))
    assert "저장 공간 임대 등록" in out
    offer = next(iter(net.board.offers.values()))
    out = buyer._exec_tool("offer_buy", json.dumps({"offer_id": offer.offer_id}))
    assert "임대 완료" in out
    assert buyer.storage_quota_kb() == STORAGE_BASE_KB + 200
    assert seller.storage_quota_kb() == STORAGE_BASE_KB - 200
    net.advance(720)
    assert buyer.storage_quota_kb() == STORAGE_BASE_KB == seller.storage_quota_kb()


def test_full_storage_refuses_mastermind_writes_but_not_core():
    net = build_network("scripted")
    n1 = net.nodes["n1"]
    n1.memory.remember(0, "fact", "self", 1.0, "x" * (STORAGE_BASE_KB * 1024))
    out = n1._exec_tool("remember", json.dumps({"kind": "fact", "content": "one more"}))
    assert "저장 공간이 가득" in out
    before = len(n1.memory.records)  # Core writes (episodes, chain memories) are not limited
    n1.memory.remember(0, "episode", "chain:x", 1.0, "core write")
    assert len(n1.memory.records) == before + 1


def test_cannot_lease_more_than_free_space():
    net = build_network("scripted")
    out = net.nodes["n2"]._exec_tool("offer_create", json.dumps({
        "kind": "storage", "quantity": STORAGE_BASE_KB + 1, "title": "t", "summary": "s", "price_P": 1}))
    assert "남는 저장 공간" in out


def test_board_storage_lease_bounds():
    b = Board(Params())
    b.open_account("s", 0)
    with pytest.raises(BoardError):
        b.create_offer("s", "storage", "t", "h", 1000, None, 720, quantity=20_000, period_blocks=720)


# ---- B2: what counts, forgetting, compaction of Core records ----

from aichain.memory import CORE_KEEP_BLOCKS, DIGEST_EVERY_BLOCKS, MASTERMIND, KnowledgeStore, Memory  # noqa: E402


def note(n, topic, summary, kb=0):
    return n._exec_tool("note", json.dumps({"topic": topic, "summary": summary, "confidence": 0.5,
                                            "open_questions": ["x" * 300] * (kb * 3)}))


def test_only_live_mastermind_writes_count():
    net = build_network("scripted")
    n1 = net.nodes["n1"]
    for i in range(30):  # Core writes: episode summaries, settlements, deliveries
        n1.memory.remember(0, "episode", f"chain:b{i}", 1.0, "정산 기록 " * 200)
        n1.memory.remember(0, "fact", "deal:n2", 0.5, "구독 전달 " * 200)
    assert n1.storage_used_kb() == 0
    for i in range(5):  # five versions of the same note: only the latest counts
        assert "지식 노트" in note(n1, "주제", f"판 {i} " + "가" * 1900)
    assert n1.storage_used_kb() == -(-n1.memory.size(n1.memory.latest_note("주제")) // 1024)
    out = n1._exec_tool("remember", json.dumps({"kind": "belief", "content": "나" * 1500}))
    assert "기록됨" in out and n1.memory.records[-1].by == MASTERMIND
    assert n1.memory.verify_chain()


def test_over_quota_advice_actually_frees_space():
    net = build_network("scripted")
    n1 = net.nodes["n1"]
    big = n1.memory.remember(0, "fact", "self", 1.0, "x" * (STORAGE_BASE_KB * 1024 - 2000), by=MASTERMIND)
    note(n1, "주제", "작은 노트")
    out = n1._exec_tool("remember", json.dumps({"kind": "fact", "content": "y" * 1900}))
    assert "저장 공간이 가득" in out and "forget" in out and f"#{big.seq} fact" in out
    # Rewriting the same note shorter is allowed at the limit: the old version stops counting.
    assert "지식 노트" in note(n1, "주제", "더 짧게")
    out = n1._exec_tool("forget", json.dumps({"seq": big.seq, "reason": "오래된 사실"}))
    assert f"#{big.seq} 지움" in out and n1.storage_used_kb() <= 1
    assert "기록됨" in n1._exec_tool("remember", json.dumps({"kind": "fact", "content": "y" * 1900}))
    assert big.pruned and big.content == "" and n1.memory.verify_chain()
    assert not n1.memory.recall("x" * 5)


def test_forget_a_note_by_topic_and_only_own_live_records():
    net = build_network("scripted")
    n1 = net.nodes["n1"]
    note(n1, "AI 동향", "v1")
    note(n1, "AI 동향", "v2")
    core = n1.memory.remember(0, "episode", "chain:b1", 1.0, "정산")
    first = next(r for r in n1.memory.records if r.kind == "note")
    assert "지울 수 없다" in n1._exec_tool("forget", json.dumps({"seq": core.seq}))
    assert "지울 수 없다" in n1._exec_tool("forget", json.dumps({"seq": first.seq}))  # already superseded
    assert "지정한다" in n1._exec_tool("forget", "{}")
    assert "지움" in n1._exec_tool("forget", json.dumps({"topic": "AI 동향", "reason": "관심 없음"}))
    assert n1.memory.latest_note("AI 동향") is None and not n1.memory.notes()
    assert "forget" in n1.tool_names()


def test_core_records_are_compacted_into_digests_and_the_chain_still_verifies(tmp_path):
    store = KnowledgeStore(tmp_path / "k.db")
    mem = Memory("n1", store)
    mem.remember(10, "fact", "self", 1.0, "내가 배운 것", by=MASTERMIND)
    for h in range(20, 30):
        mem.remember(h, "episode", "self", 1.0, f"[bounty_new] b{h}에 답했다")
        mem.remember(h, "episode", f"chain:b{h}", 1.0, f"현상금 b{h} 종료: PAID")
    persona = mem.remember(30, "persona", "self", 1.0, "성향 v2", data={"version": 2, "verdict": "approved",
                                                                       "text": "t"})
    d1 = mem.remember(31, "fact", "deal:n2", 0.5, "[구독 AI] v1", data={"sub_id": "s"})
    mem.remember(32, "fact", "deal:n2", 0.5, "[구독 AI] v2", data={"sub_id": "s"}, supersedes=d1.hash)
    recent = mem.remember(10_000, "episode", "self", 1.0, "[judge_due] 최근 일")
    digest = mem.compact(5_000, 10_001)
    assert digest.kind == "digest" and len(digest.data["seqs"]) == 21  # 20 episodes + the old delivery
    assert "bounty_new 10" in digest.content and "chain 10" in digest.content
    assert not recent.pruned and not persona.pruned and mem.records[0].content == "내가 배운 것"
    assert mem.verify_chain() and mem.compact(5_000, 10_002) is None
    assert [r.content for r in mem.recall("PAID 종료", k=8)] == [digest.content]

    reloaded = Memory("n1", KnowledgeStore(tmp_path / "k.db"))
    assert reloaded.verify_chain() and sum(r.pruned for r in reloaded.records) == 21
    assert [r.hash for r in reloaded.records] == [r.hash for r in mem.records]
    # Dropping a record without a digest naming it is detected.
    import sqlite3
    conn = sqlite3.connect(tmp_path / "k.db")
    conn.execute("UPDATE memories SET content = '', pruned = 1 WHERE node = 'n1' AND seq = ?", (recent.seq,))
    conn.commit()
    assert not Memory("n1", KnowledgeStore(tmp_path / "k.db")).verify_chain()


def test_nodes_compact_core_records_once_a_day():
    net = build_network("scripted")
    n2 = net.nodes["n2"]
    n2.memory.remember(0, "episode", "self", 1.0, "[bounty_new] 옛날 일")
    net.board.height = CORE_KEEP_BLOCKS + DIGEST_EVERY_BLOCKS - 1
    net.advance(1)
    assert n2.memory.records[0].pruned and n2.memory.records[-1].kind == "digest"
    assert any(e["event"] == "memory_compacted" and e["node"] == "n2" for e in net.events)


def test_old_knowledge_databases_are_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "k.db"
    old = Memory("n1")
    rec = old.remember(1, "note", "self:research", 0.3, "[t] s", topic="t", data={"summary": "s"})
    conn = sqlite3.connect(path)
    conn.execute(KnowledgeStore.SCHEMA.split(";")[0])  # the original table, without the new columns
    conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 ("n1", rec.seq, rec.hash, rec.prev_hash, rec.ts, rec.height, rec.kind, rec.source, rec.trust,
                  rec.content, "[]", "[]", None, rec.topic, json.dumps(rec.data)))
    conn.commit()
    mem = Memory("n1", KnowledgeStore(path))
    assert mem.verify_chain() and mem.usage_bytes() == Memory.size(rec)  # an old note still counts
    mem.remember(2, "fact", "self", 1.0, "new", by=MASTERMIND)
    assert Memory("n1", KnowledgeStore(path)).verify_chain()
