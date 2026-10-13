"""Gap review B3: the block loop skips finished objects, the chain writes signed state snapshots,
replay starts from the newest one that verifies, and block headers carry a state root only every
ROOT_CHECK_EVERY blocks."""

import json
import random

import pytest

from aichain import board as board_mod
from aichain.board import Board, BoardError, BountyState, Params, commitment_of
from aichain.chain import store as chain_store
from aichain.chain.store import ChainStore, ReplayMismatch, attach, replay
from aichain.common import sha256_hex

P = 1000
NAMES = ["n1", "n2", "n3", "n4"]


def params():
    return Params(commit_blocks=12, reveal_blocks=12, judge_blocks=12, dispute_blocks=20,
                  dispute_resolve_blocks=720, epoch_blocks=60, vote_blocks=720, timelock_blocks=360,
                  proposal_deposit_mP=5 * P)


def setup(b):
    for n in NAMES:
        b.open_account(n, 2000 * P)
    b.open_account("operator", 0)


def step(b, rnd):
    """One random transaction from most kinds the board has, then a few blocks."""
    who, other = rnd.sample(NAMES, 2)
    op = rnd.randrange(16)
    try:
        if op == 0:
            b.create_bounty(who, f"q{b.height}", rnd.randint(1, 5) * P, ["t"])
        elif op == 1:
            open_b = [x for x in b.bounties.values() if x.state == BountyState.OPEN and x.asker != who]
            if open_b:
                b.commit_answer(who, rnd.choice(open_b).bounty_id, commitment_of("h", "s", who))
        elif op == 2:
            for a in list(b.answers.values()):
                if not a.revealed and b.bounties[a.bounty_id].state == BountyState.REVEAL:
                    b.reveal_answer(a.author, a.answer_id, "h", "s")
        elif op == 3:
            judging = [x for x in b.bounties.values() if x.state == BountyState.JUDGING]
            if judging:
                x = rnd.choice(judging)
                revealed = [a for a in x.answers if b.answers[a].revealed]
                if revealed:
                    b.accept_answer(x.asker, x.bounty_id, rnd.choice(revealed))
        elif op == 4:
            o = b.create_offer(who, "info", "t", "terms", 2 * P, sha256_hex(f"{who}{b.height}"), 720)
            pu = b.buy_offer(other, o.offer_id)
            if rnd.random() < 0.8:
                b.deliver_offer(who, pu.purchase_id, "c")
                if rnd.random() < 0.5:
                    b.confirm_delivery(other, pu.purchase_id, rnd.random() < 0.7, "r")
        elif op == 5:
            loan = b.create_loan_request(who, 5 * P, 500, rnd.randint(0, 8) * P, 720, "t")
            if rnd.random() < 0.6:
                b.fund_loan(other, loan.loan_id)
        elif op == 6:
            b.deposit_savings(who, rnd.randint(1, 20) * P, 720)
        elif op == 7:
            o = b.create_offer(who, "subscription", "s", "terms", 1 * P, None, 2000, period_blocks=720)
            sub = b.subscribe(other, o.offer_id)
            if rnd.random() < 0.5:
                b.sub_deliver(who, sub.sub_id, "c")
        elif op == 8:
            b.create_tool_request(who, "tool", "spec", "tests", 2 * P, 150)
        elif op == 9:
            tid = f"{who}.t{b.height}"
            b.publish_tool(who, tid, 1, "code", 1 * P, 100)
            b.license_tool(other, tid, 1)
            b.pay_tool_use(other, tid, 1, 2)
        elif op == 10:
            b.rate(who, other, rnd.randint(-2, 2), f"ctx{b.height}")
        elif op == 11:
            b.transfer(who, other, rnd.randint(1, 3) * P, "m")
        elif op == 12:
            if not any(p.state == "VOTING" for p in b.proposals.values()):
                prop = b.create_proposal(who, "param", "t", "spec", [{"key": "commit_blocks", "to": 24}])
                for v in rnd.sample(NAMES, 3):
                    if v != who:
                        b.vote(v, prop.proposal_id, rnd.choice(["yes", "no"]), "r")
        elif op == 13:
            paid = [a for a in b.answers.values()
                    if b.bounties[a.bounty_id].state in (BountyState.AUTO_SPLIT, BountyState.PAID)]
            if paid:
                a = rnd.choice(paid)
                b.open_dispute(rnd.choice([n for n in NAMES if n != a.author]), "answer", a.answer_id, "c")
        elif op == 14:
            for d in [d for d in b.disputes.values() if d.state == "open"][:1]:
                b.resolve_dispute("root", d.dispute_id, rnd.choice(["uphold", "reject", "void"]), "r")
        else:
            for n in NAMES:
                b.heartbeat(n, b.height // 720)
    except BoardError:
        pass
    b.advance(rnd.randint(1, 25))


def walk_everything(self, name):
    """The block loop as it was before the index: every object ever created, in order."""
    return list(getattr(self, name).values())


def test_block_loop_index_matches_walking_everything(monkeypatch):
    rnd_a, rnd_b = random.Random(11), random.Random(11)
    fast, full = Board(params()), Board(params())
    setup(fast), setup(full)
    roots = {}
    for i in range(1500):
        step(fast, rnd_a)
        if i % 10 == 0:
            roots[i] = fast.state_root()
    monkeypatch.setattr(Board, "_live", walk_everything)
    for i in range(1500):
        step(full, rnd_b)
        if i % 10 == 0:
            assert full.state_root() == roots[i], f"step {i}"
    assert full.state_root() == fast.state_root()
    kinds = {name: len(getattr(fast, name)) for name in board_mod.LIVE_IN_BLOCK_LOOP}
    assert all(kinds.values()), kinds  # the run reached every collection the loop walks
    assert sum(map(len, fast._live_index.values())) < sum(kinds.values()) // 2


def test_snapshot_round_trip_continues_identically():
    rnd = random.Random(5)
    b = Board(params())
    setup(b)
    for _ in range(300):
        step(b, rnd)
    data = json.loads(json.dumps(b.to_snapshot()))
    copy = Board(Params())
    copy.restore(data)
    assert copy.state_root() == b.state_root()
    assert json.loads(json.dumps(copy.to_snapshot())) == data
    r1, r2 = random.Random(9), random.Random(9)
    for _ in range(200):
        step(b, r1)
        step(copy, r2)
        assert copy.state_root() == b.state_root()


def test_snapshot_refuses_fields_it_does_not_know():
    b = Board(params())
    b.new_thing = {}
    with pytest.raises(BoardError, match="new_thing"):
        b.to_snapshot()


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(chain_store, "ROOT_CHECK_EVERY", 20)
    monkeypatch.setattr(chain_store, "SNAPSHOT_EVERY", 100)


def live_chain(db, steps=200, seed=3):
    b = Board(params())
    attach(b, ChainStore(db))
    setup(b)
    rnd = random.Random(seed)
    for _ in range(steps):
        step(b, rnd)
    return b


def test_replay_starts_from_the_newest_snapshot(tmp_path, small):
    db = tmp_path / "chain.db"
    live = live_chain(db)
    store = ChainStore(db)
    heights = [h["height"] for h, _ in store.snapshots()]
    assert len(heights) == chain_store.SNAPSHOTS_KEPT and heights == sorted(heights, reverse=True)
    assert all(h % 100 == 0 for h in heights)
    headers = [e[3] for e in store.entries() if e[0] == "block"]
    assert all(bool(h["state_root"]) == (h["height"] % 20 == 0) for h in headers)

    quick = Board(params())
    stats = replay(quick, store)
    assert stats["snapshot_height"] == heights[0] and stats["blocks"] < 100
    full = Board(params())
    stats_full = replay(full, ChainStore(db), from_snapshot=False)
    assert "snapshot_height" not in stats_full and stats_full["blocks"] == live.height
    assert quick.state_root() == full.state_root() == live.state_root()

    # the restarted chain keeps going and the next start still agrees
    again = Board(params())
    attach(again, ChainStore(db))
    rnd = random.Random(4)
    for _ in range(60):
        step(again, rnd)
    third = Board(params())
    attach(third, ChainStore(db))
    assert third.state_root() == again.state_root()


def _edit(db, sql, args=()):
    import sqlite3
    with sqlite3.connect(db) as c:
        c.execute(sql, args)


def test_bad_snapshots_fall_back(tmp_path, small):
    db = tmp_path / "chain.db"
    live = live_chain(db)
    (newest, _), (second, _), (oldest, _) = ChainStore(db).snapshots()
    # tampered data: the hash check fails, the next one is used
    import zlib
    data = json.loads(ChainStore(db).snapshot_data(newest["height"]))
    data["accounts"][0]["balance_mP"] += 1
    _edit(db, "UPDATE snapshots SET data = ? WHERE height = ?",
          (zlib.compress(json.dumps(data).encode()), newest["height"]))
    # a re-signed header cannot be made without the root key; a forged one fails the signature
    forged = {**second, "nonces": {**second["nonces"], "n1": 1}}
    _edit(db, "UPDATE snapshots SET header = ? WHERE height = ?", (json.dumps(forged), second["height"]))
    b = Board(params())
    stats = replay(b, ChainStore(db))
    assert stats["snapshot_height"] == oldest["height"]
    assert len(stats["snapshots_rejected"]) == 2
    assert b.state_root() == live.state_root()

    _edit(db, "UPDATE snapshots SET sig = '00' WHERE height = ?", (oldest["height"],))
    b = Board(params())
    stats = replay(b, ChainStore(db))
    assert "snapshot_height" not in stats and len(stats["snapshots_rejected"]) == 3
    assert b.state_root() == live.state_root()


def test_snapshot_must_sit_on_its_block(tmp_path, small):
    db = tmp_path / "chain.db"
    live_chain(db)
    (newest, _), *_ = ChainStore(db).snapshots()
    _edit(db, "DELETE FROM log WHERE seq > ?", (newest["seq"] - 1,))  # the log lost that block
    b = Board(params())
    stats = replay(b, ChainStore(db))
    assert stats["snapshot_height"] != newest["height"]
    assert any("not in the log" in r for r in stats["snapshots_rejected"])


def test_tampered_log_after_snapshot_is_still_caught(tmp_path, small):
    db = tmp_path / "chain.db"
    live = live_chain(db)
    (newest, _), *_ = ChainStore(db).snapshots()
    live.transfer("n1", "n2", 1234, "m")
    live.advance(1)
    _edit(db, "UPDATE log SET data = replace(data, '1234', '9234') WHERE kind = 'tx' AND seq > ?",
          (newest["seq"],))
    with pytest.raises(ReplayMismatch, match="id does not match"):
        replay(Board(params()), ChainStore(db))


def test_verify_tool(tmp_path, small, capsys, monkeypatch):
    import aichain.sim.factory as factory
    from aichain.chain import verify
    db = tmp_path / "chain.db"
    live = live_chain(db, steps=80)
    # a snapshot carries the parameters; only a full replay depends on the starting ones
    assert verify.main([str(db), "--periods", "design", "--full"]) == 3
    assert verify.main([str(db), "--periods", "design"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["snapshot_height"] and out["escrow_ok"] and out["supply_ok"]
    monkeypatch.setattr(factory, "SHORT_PARAMS", params())  # what this chain started from
    assert verify.main([str(db), "--full"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "snapshot_height" not in out and out["height"] == live.height
