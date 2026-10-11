import random
import sqlite3

import pytest

from aichain.board import Board, BoardError, BountyState, Params, commitment_of
from aichain.chain.store import CHECKPOINT_EVERY, ChainStore, ReplayMismatch, attach, replay
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS, build_network

P = SHORT_PARAMS
CYCLE = P.commit_blocks + P.reveal_blocks + P.judge_blocks + P.dispute_blocks + 2


def test_full_restart_mid_bounty_matches_uninterrupted_run(tmp_path):
    # Reference run, no restarts.
    ref = build_network("scripted")
    ref.bridge_question("n1", "t", **DEFAULT_QUESTION)
    ref.run_until_idle()
    ref.advance(CYCLE)

    db = tmp_path / "chain.db"
    net = build_network("scripted", chain=ChainStore(db))
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()  # bounty created, two answers committed (sealed, not revealed)
    net.advance(5)
    assert len(ChainStore(db).sealed_for("n2")) == 1

    # Restart 1: answers still sealed; the Core must reveal them from the stored copy.
    net = build_network("scripted", chain=ChainStore(db))
    assert net.chain_stats["height"] == 5 and net.chain_stats["transactions"] >= 7
    net.advance(P.commit_blocks + P.reveal_blocks - 5 - 1)  # stop just before JUDGING... then
    net.advance(1)
    (b,) = net.board.bounties.values()
    assert b.state in (BountyState.JUDGING, BountyState.PAID)

    # Restart 2 while judging is due: the duty is re-raised and the asker judges.
    net = build_network("scripted", chain=ChainStore(db))
    net.run_until_idle()
    net.advance(CYCLE)
    (b,) = net.board.bounties.values()
    assert b.state == BountyState.FINAL and b.outcome == BountyState.PAID
    assert net.board.answers[b.accepted].author == "n2"
    for nid in ("n1", "n2", "n3", "n4"):
        assert net.board.accounts[nid].balance_mP == ref.board.accounts[nid].balance_mP
    assert net.board.total_supply_check()
    assert ChainStore(db).sealed_for("n2") == {}


def test_replay_reproduces_state_root(tmp_path):
    db = tmp_path / "c.db"
    for seed in range(5):
        rng = random.Random(seed)
        store = ChainStore(tmp_path / f"c{seed}.db")
        live = Board(Params(commit_blocks=12, reveal_blocks=6, judge_blocks=10, dispute_blocks=20))
        attach(live, store)
        for n in ("a", "b", "c"):
            live.open_account(n, 100_000)
        for _ in range(150):
            try:
                op = rng.random()
                if op < 0.2:
                    live.create_bounty(rng.choice("abc"), "q", rng.randint(1, 20) * 1000, [])
                elif op < 0.45 and live.bounties:
                    live.commit_answer(rng.choice("abc"), rng.choice(list(live.bounties)),
                                       commitment_of("h", "s", "x"))
                elif op < 0.55 and live.answers:
                    a = live.answers[rng.choice(list(live.answers))]
                    b = live.bounties[a.bounty_id]
                    live.reject_answer(b.asker, b.bounty_id, a.answer_id, "no")
                else:
                    live.advance(rng.randint(1, 30))
            except BoardError:
                pass
        rebuilt = Board(live.params)
        replay(rebuilt, store)
        rebuilt.advance(live.height - rebuilt.height)
        assert rebuilt.state_root() == live.state_root(), seed


def _signed_chain(tmp_path):
    db = tmp_path / "c.db"
    store = ChainStore(db)
    board = Board(Params())
    attach(board, store)
    board.open_account("a", 100_000)
    board.open_account("b", 100_000)
    board.create_bounty("a", "q", 5_000, [])
    board.advance(CHECKPOINT_EVERY)
    return db


def _edit(db, sql):
    conn = sqlite3.connect(db)
    conn.execute(sql)
    conn.commit()


def test_tampered_transaction_is_detected(tmp_path):
    db = _signed_chain(tmp_path)
    _edit(db, "UPDATE log SET data = replace(data, '5000', '9000') WHERE data LIKE '%create_bounty%'")
    with pytest.raises(ReplayMismatch, match="id does not match"):
        replay(Board(Params()), ChainStore(db))


def test_forged_signature_and_reordered_nonce_are_detected(tmp_path):
    db = _signed_chain(tmp_path)
    _edit(db, "UPDATE log SET sig = substr(sig, 1, 10) || 'ff' || substr(sig, 13) "
              "WHERE kind = 'tx' AND data LIKE '%create_bounty%'")
    with pytest.raises(ReplayMismatch, match="bad signature"):
        replay(Board(Params()), ChainStore(db))
    db2 = _signed_chain(tmp_path / "x")
    _edit(db2, "DELETE FROM log WHERE kind = 'tx' AND data LIKE '%\"account_id\": \"a\"%'")
    with pytest.raises(ReplayMismatch):
        replay(Board(Params()), ChainStore(db2))


def test_blocks_are_signed_linked_and_include_their_transactions(tmp_path):
    db = _signed_chain(tmp_path)
    entries = ChainStore(db).entries()
    blocks = [e for e in entries if e[0] == "block"]
    assert len(blocks) == CHECKPOINT_EVERY
    assert all(b[3]["prev_hash"] == a[2] for a, b in zip(blocks, blocks[1:]))
    assert blocks[0][3]["tx_root"] != blocks[1][3]["tx_root"]  # the first block carries 3 txs
    # A block removed from the middle breaks the chain.
    _edit(db, f"DELETE FROM log WHERE kind = 'block' AND height = 50")
    with pytest.raises(ReplayMismatch, match="does not extend"):
        replay(Board(Params()), ChainStore(db))


def test_wrong_root_key_is_refused(tmp_path):
    from aichain.chain.crypto import KeyStore

    db = _signed_chain(tmp_path)
    with pytest.raises(ReplayMismatch, match="root key"):
        attach(Board(Params()), ChainStore(db), KeyStore(tmp_path / "other-keys"))


def test_transactions_are_signed_by_the_party_they_act_for(tmp_path):
    from aichain.chain.crypto import verify

    db = _signed_chain(tmp_path)
    txs = [e for e in ChainStore(db).entries() if e[0] == "tx"]
    bounty = next(t for t in txs if t[3]["type"] == "create_bounty")
    assert bounty[3]["from"] == "a" and bounty[3]["nonce"] == 1
    board = Board(Params())
    replay(board, ChainStore(db))
    assert verify(board.accounts["a"].pubkey, "tx", bounty[2], bounty[4])
    assert not verify(board.accounts["b"].pubkey, "tx", bounty[2], bounty[4])


def test_content_survives_restart(tmp_path):
    db = tmp_path / "c.db"
    net = build_network("scripted", chain=ChainStore(db))
    h = net.put_content({"title": "x"}, author="n1")
    again = build_network("scripted", chain=ChainStore(db))
    assert again.content.get(h) == {"title": "x"} and again.content_author[h] == "n1"
    assert set(again.board.accounts) == {"n1", "n2", "n3", "n4", "operator"}
    assert again.board.issued_mP == 4_000_000  # genesis grants not repeated
