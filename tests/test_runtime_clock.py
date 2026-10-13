"""B7: blocks follow the wall clock while episodes wait on the model; missed heights are caught up."""

import json
import threading
import time

from aichain.bridge.runtime import BoardLock, NetworkRuntime
from aichain.node import QueuedEvent
from aichain.sim.brains import persona_brain
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import build_network

BLOCK = 0.02


def wait_until(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_blocks_keep_coming_while_an_episode_waits_on_the_model():
    seen = {}

    def slow(request):
        if request.node_id == "n1" and "bridge_question" in (request.messages[-1].content or ""):
            seen["start"] = net.board.height
            time.sleep(0.5)  # a slow provider
            seen["end"] = net.board.height
        return persona_brain(request)

    net = build_network("scripted", brain=slow)
    net.audit_interval_blocks = 5  # something the block thread logs on its own
    runtime = NetworkRuntime(net, block_seconds=BLOCK)
    entries = []
    net.listeners.append(lambda e: entries.append((threading.current_thread().name, runtime.board_lock.held(), e)))
    runtime.start()
    try:
        runtime.submit("c", **DEFAULT_QUESTION)
        assert wait_until(lambda: "end" in seen)
        # While n1 waited on the model, an admin action ran right away on the block thread.
        assert runtime.call(lambda n: n.board.height, timeout=2) >= seen["end"]
        assert wait_until(lambda: len(net.board.bounties) == 1)
    finally:
        runtime.stop()
    assert seen["end"] - seen["start"] >= 10  # ~25 blocks went by during the 0.5 s call
    assert all(held for _, held, _ in entries)  # every log entry was made holding the board lock
    block_entries = [e for name, _, e in entries if name == "aichain-blocks"]
    assert any(e["event"] == "audit" for e in block_entries)
    assert not any("episode" in e for e in block_entries)  # blocks made meanwhile are not part of it
    assert any(e.get("episode") for name, _, e in entries if name == "aichain-episodes")
    assert runtime.errors == []


def test_missed_heights_are_caught_up_after_a_stall():
    net = build_network("scripted")
    runtime = NetworkRuntime(net, block_seconds=BLOCK)
    runtime.start()
    try:
        assert wait_until(lambda: net.board.height >= 3)
        t0, h0 = time.monotonic(), net.board.height
        runtime.call(lambda n: time.sleep(0.4))  # something slow holds the board on the block thread
        assert wait_until(lambda: net.board.height >= h0 + (time.monotonic() - t0) / BLOCK - 3)
    finally:
        runtime.stop()
    caught = [e for e in net.events if e["event"] == "blocks_caught_up"]
    assert caught and max(e["blocks"] for e in caught) >= 10


def test_board_lock_is_fifo_and_hands_over():
    lock, order = BoardLock(), []
    lock.acquire()
    waiters = [threading.Thread(target=lambda i=i: (lock.acquire(), order.append(i), lock.release()))
               for i in range(3)]
    for t in waiters:
        t.start()
        time.sleep(0.05)  # each asks in turn
    assert lock.held() and not order
    lock.release()
    for t in waiters:
        t.join(2)
    assert order == [0, 1, 2] and not lock.held()


def test_an_approval_executed_on_a_block_does_not_act_in_a_running_episode():
    net = build_network("scripted")
    n1 = net.nodes["n1"]
    ev = QueuedEvent(3, 0, "bridge_question", {"ref": "q_x", "max_reward_P": 1, "client": "c",
                                                 **DEFAULT_QUESTION})
    n1.current_event = ev  # n1 is waiting on the model in a Bridge question episode
    out = n1.execute_approval(json.dumps({"question": DEFAULT_QUESTION, "reward_P": 5, "tags": []}), "ask")
    assert "현상금 생성" in out and n1.current_event is ev
    assert not any(e["event"] == "bridge_linked" for e in net.events)  # not tied to that question


def test_only_the_episode_thread_lets_go_of_the_board_while_waiting_on_a_provider():
    net = build_network("scripted")
    runtime = NetworkRuntime(net, block_seconds=BLOCK)
    runtime.start()
    try:
        def admin(n):  # an admin action that ends up calling a provider keeps the block atomic
            with n.gateway.call_guard():
                return runtime.board_lock.held()
        assert runtime.call(admin, timeout=2)
    finally:
        runtime.stop()
    assert net.gateway.call_guard is None and net.inline_episodes
