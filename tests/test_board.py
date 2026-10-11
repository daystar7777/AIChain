import random

import pytest

from aichain.board import Board, BoardError, BountyState, Params, commitment_of

P = 1000


def make_board(**kw):
    b = Board(Params(commit_blocks=12, reveal_blocks=6, judge_blocks=10, dispute_blocks=20, **kw))
    for n in ("asker", "a1", "a2", "a3"):
        b.open_account(n, 1000 * P)
    return b


def commit(board, author, bounty_id, body_hash="h", salt="s"):
    return board.commit_answer(author, bounty_id, commitment_of(body_hash, salt, author))


def to_judging(board, bounty, reveal=("a1", "a2")):
    answers = {board.answers[a].author: board.answers[a] for a in bounty.answers}
    board.advance(bounty.commit_until - board.height)
    assert bounty.state == BountyState.REVEAL
    for author in reveal:
        board.reveal_answer(author, answers[author].answer_id, "h", "s")
    board.advance(bounty.reveal_until - board.height)
    return answers


def test_paid_path_locks_then_final():
    board = make_board()
    b = board.create_bounty("asker", "q", 20 * P, ["code"])
    assert board.account("asker").available_mP == 980 * P
    commit(board, "a1", b.bounty_id)
    commit(board, "a2", b.bounty_id)
    answers = to_judging(board, b)
    assert b.state == BountyState.JUDGING
    events = board.accept_answer("asker", b.bounty_id, answers["a1"].answer_id)
    assert b.state == BountyState.PAID and events[0].payload["outcome"] == "PAID"
    a1 = board.account("a1")
    assert a1.balance_mP == 1020 * P and a1.locked_mP == 20 * P
    assert board.account("a2").balance_mP == 1000 * P  # bond returned on reveal
    board.advance(20)
    assert b.state == BountyState.FINAL and a1.locked_mP == 0
    assert board.total_supply_check()


def test_auto_split_burns_five_percent():
    board = make_board()
    b = board.create_bounty("asker", "q", 20 * P, [])
    commit(board, "a1", b.bounty_id)
    commit(board, "a2", b.bounty_id)
    commit(board, "a3", b.bounty_id)
    answers = to_judging(board, b)  # a3 never reveals
    board.reject_answer("asker", b.bounty_id, answers["a2"].answer_id, "wrong")
    board.advance(b.judge_until - board.height)
    assert b.state == BountyState.AUTO_SPLIT
    assert board.account("a1").balance_mP == 1019 * P  # 95% of 20P to the only survivor
    assert board.account("a3").balance_mP == 999_500  # unrevealed bond burned
    assert board.burned_mP == 1 * P + 500
    assert any(e["kind"] == "judge_timeout" for e in board.rep_events)
    assert board.total_supply_check()


def test_all_rejected_refunds_95_percent():
    board = make_board()
    b = board.create_bounty("asker", "q", 20 * P, [])
    commit(board, "a1", b.bounty_id)
    answers = to_judging(board, b, reveal=("a1",))
    board.reject_answer("asker", b.bounty_id, answers["a1"].answer_id, "wrong")
    board.advance(b.judge_until - board.height)
    assert b.state == BountyState.REFUNDED
    assert board.account("asker").balance_mP == 999 * P
    assert board.total_supply_check()


def test_expired_without_answers_refunds_fully():
    board = make_board()
    b = board.create_bounty("asker", "q", 20 * P, [])
    board.advance(12)
    assert b.state == BountyState.EXPIRED
    assert board.account("asker").balance_mP == 1000 * P
    assert board.account("asker").locked_mP == 0


def test_expired_when_nobody_reveals():
    board = make_board()
    b = board.create_bounty("asker", "q", 20 * P, [])
    commit(board, "a1", b.bounty_id)
    board.advance(18)
    assert b.state == BountyState.EXPIRED
    assert board.account("asker").balance_mP == 1000 * P
    assert board.burned_mP == 500


@pytest.mark.parametrize("bad", [
    lambda bd, b: commit(bd, "asker", b.bounty_id),  # self answer
    lambda bd, b: (commit(bd, "a1", b.bounty_id), commit(bd, "a1", b.bounty_id)),  # twice
    lambda bd, b: bd.create_bounty("asker", "q", 500, []),  # below min reward
    lambda bd, b: bd.create_bounty("asker", "q", 2000 * P, []),  # more than balance
    lambda bd, b: bd.accept_answer("asker", b.bounty_id, "a_x"),  # not JUDGING
])
def test_protocol_rejections(bad):
    board = make_board()
    b = board.create_bounty("asker", "q", 20 * P, [])
    with pytest.raises(BoardError):
        bad(board, b)


def test_reveal_must_match_commitment_and_only_asker_judges():
    board = make_board()
    b = board.create_bounty("asker", "q", 20 * P, [])
    a = commit(board, "a1", b.bounty_id)
    board.advance(12)
    with pytest.raises(BoardError):
        board.reveal_answer("a1", a.answer_id, "other", "s")
    board.reveal_answer("a1", a.answer_id, "h", "s")
    board.advance(6)
    with pytest.raises(BoardError):
        board.accept_answer("a2", b.bounty_id, a.answer_id)


def test_locked_winnings_cannot_be_spent():
    board = make_board()
    board.open_account("poor", 0)
    b = board.create_bounty("asker", "q", 20 * P, [])
    board.open_account("x", 1 * P)
    a = commit(board, "x", b.bounty_id)
    board.advance(12)
    board.reveal_answer("x", a.answer_id, "h", "s")
    board.advance(6)
    board.accept_answer("asker", b.bounty_id, a.answer_id)
    with pytest.raises(BoardError):
        board.create_bounty("x", "q2", 10 * P, [])  # 21P balance but 20P locked


def test_random_runs_conserve_supply():
    """Property: issued == balances + escrow + burned, and balance >= locked >= 0."""
    for seed in range(30):
        rng = random.Random(seed)
        board = make_board()
        names = ["asker", "a1", "a2", "a3"]
        for _ in range(200):
            op = rng.random()
            try:
                if op < 0.15:
                    board.create_bounty(rng.choice(names), "q", rng.randint(1, 50) * P, [])
                elif op < 0.4 and board.bounties:
                    commit(board, rng.choice(names), rng.choice(list(board.bounties)))
                elif op < 0.55 and board.answers:
                    a = board.answers[rng.choice(list(board.answers))]
                    board.reveal_answer(a.author, a.answer_id, "h", "s")
                elif op < 0.65 and board.answers:
                    a = board.answers[rng.choice(list(board.answers))]
                    bnty = board.bounties[a.bounty_id]
                    if rng.random() < 0.5:
                        board.accept_answer(bnty.asker, bnty.bounty_id, a.answer_id)
                    else:
                        board.reject_answer(bnty.asker, bnty.bounty_id, a.answer_id, "no")
                else:
                    board.advance(rng.randint(1, 8))
            except BoardError:
                pass
            assert board.total_supply_check(), seed
            for acct in board.accounts.values():
                assert acct.balance_mP >= acct.locked_mP >= 0, seed
        board.advance(200)
        assert board.escrow_mP == 0, seed
        assert all(b.state in (BountyState.FINAL, BountyState.EXPIRED)
                   for b in board.bounties.values()), seed
