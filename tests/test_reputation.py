import math

from aichain.board import Board, BoardError, Params, commitment_of
from aichain.reputation import E6, compute, decay_e6, for_board, log10_e6
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import SHORT_PARAMS, build_network

import pytest


def test_integer_math_matches_floats_closely():
    for x in (1.0, 1.5, 2.0, 9.99, 10.0, 123.0, 1000.0, 0.5):
        assert abs(log10_e6(int(x * E6)) / E6 - math.log10(x)) < 2e-3
    assert decay_e6(0) == E6 and decay_e6(120_960) == E6 // 2
    assert decay_e6(241_920) == E6 // 4
    assert abs(decay_e6(60_480) / E6 - 0.5 ** 0.5) < 0.07  # linear interpolation inside a half-life


def test_new_node_starts_at_045():
    r = compute([], "x", 0)
    assert r.value_e6 == 450_000
    assert r.terms_e6 == {"A": 500_000, "V": 0, "J": E6, "T": 500_000, "W": 500_000, "U": 0, "S": 500_000}


def test_events_move_reputation_the_right_way():
    ev = lambda kind, h=0, **x: {"account": "x", "kind": kind, "height": h, **x}
    base = compute([], "x", 0).value_e6
    good = compute([ev("answer_accepted", value_mP=100_000)] * 3, "x", 0)
    assert good.value_e6 > base and good.terms_e6["V"] > 0
    bad = compute([ev("answer_rejected")] * 3, "x", 0)
    assert bad.value_e6 < base
    lazy = compute([ev("judge_timeout")] * 2, "x", 0, bounties_created=2)
    assert lazy.terms_e6["J"] == E6 // 2
    # Old events count less: a rejection a month ago hurts less than one today.
    assert compute([ev("answer_rejected", 0)], "x", 518_400).value_e6 > compute(
        [ev("answer_rejected", 518_400)], "x", 518_400).value_e6
    # 1000P of income maxes out V.
    assert compute([ev("answer_accepted", value_mP=1_000_000)], "x", 0).terms_e6["V"] == E6


def test_reputation_is_deterministic_and_part_of_the_state_root():
    a, b = Board(Params()), Board(Params())
    for board in (a, b):
        board.open_account("x", 1000)
        board.advance(720)
        board.heartbeat("x", 1)
    assert for_board(a, "x").value_e6 == for_board(b, "x").value_e6
    assert a.state_root() == b.state_root()
    with pytest.raises(BoardError):
        a.heartbeat("x", 1)  # one per epoch
    with pytest.raises(BoardError):
        a.heartbeat("x", 5)  # not the current epoch


def test_network_uses_computed_reputation():
    net = build_network("scripted")
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()
    p = SHORT_PARAMS
    net.advance(p.commit_blocks + p.reveal_blocks + p.judge_blocks + p.dispute_blocks + 2)
    assert net.reputation("n2").value > net.reputation("n3").value  # n2's answer was accepted
    assert net.reputation("n4").value == pytest.approx(0.45, abs=0.01)  # answered, not accepted, not rejected
    prompt = net.gateway.adapters["scripted"].requests[-1].messages[1].content
    assert "다른 노드 평판" in prompt and "(프로토타입: 고정값)" not in prompt


def test_heartbeats_every_epoch():
    net = build_network("scripted")
    net.advance(720 * 2)
    assert sum(1 for e in net.board.rep_events if e["kind"] == "heartbeat") == 8
    assert net.reputation("n1").terms_e6["U"] > 0
