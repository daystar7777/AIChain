"""Running for months: bounded memory, rotating files, a full disk, halts and the watchdog."""

import errno
import os
import time

from aichain.bridge import runtime as runtime_mod
from aichain.bridge.runtime import NetworkRuntime
from aichain.gateways.web import FakeWebProvider, SearchGateway, WebLimits
from aichain.ops import DailyTrace, prune_files
from aichain.sim.demo import DEFAULT_QUESTION
from aichain.sim.factory import build_network
from aichain.sim.network import Network


def ask(net):
    net.bridge_question("n1", "t", **DEFAULT_QUESTION)
    net.run_until_idle()


# ---- bounded in-memory history ----

def test_network_history_is_a_ring_buffer(monkeypatch):
    monkeypatch.setattr(Network, "EVENTS_KEPT", 5)
    monkeypatch.setattr(Network, "EPISODES_KEPT", 2)
    net = build_network("scripted")
    seen = []
    net.listeners.append(seen.append)
    ask(net)
    net.advance(30)
    assert len(seen) > 5 and list(net.events) == seen[-5:]
    assert len(net.episodes) == 2 and net.gateway.log.maxlen == 1000


def test_manager_outbox_is_bounded_and_ids_grow_across_restarts(monkeypatch):
    monkeypatch.setattr(Network, "OUTBOX_KEPT", 3)
    net = build_network("scripted")
    ids = [net.manager_send("n1", f"m{i}")["id"] for i in range(5)]
    assert ids == sorted(set(ids)) and len(net.manager_outbox) == 3
    assert [m["text"] for m in net.manager_outbox] == ["m2", "m3", "m4"]
    while int(time.time() * 1000) <= ids[-1]:  # a restart takes seconds (chain replay)
        time.sleep(0.001)
    again = build_network("scripted")  # a restarted process: the Telegram bridge saw ids[-1]
    assert again.manager_send("core", "after restart")["id"] > ids[-1]


def test_web_cache_expires_and_is_capped():
    now = [1000.0]
    pages = {f"https://e.org/{i}": {"title": "t", "text": f"page {i}"} for i in range(6)}
    gw = SearchGateway(FakeWebProvider(pages=pages), WebLimits(fetch_per_hour=100, fetch_ttl=60, cache_entries=3),
                       clock=lambda: now[0])
    for i in range(3):
        gw.fetch("n1", f"https://e.org/{i}", 100)
    assert gw.fetch("n1", "https://e.org/0", 100)["cached"] is True
    now[0] += 61  # all three expire
    gw.fetch("n1", "https://e.org/3", 100)
    assert len(gw._cache) == 1  # the expired ones went first
    for i in (4, 5, 0, 1):
        gw.fetch("n1", f"https://e.org/{i}", 100)
    assert len(gw._cache) == 3 and {k[2] for k in gw._cache} == {"100|https://e.org/5", "100|https://e.org/0",
                                                                 "100|https://e.org/1"}
    now[0] += 61
    assert "cached" not in gw.fetch("n1", "https://e.org/1", 100)  # expired hit: fetched again


def test_bridge_question_records_are_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_mod, "MAX_QUESTIONS", 3)
    net = build_network("scripted")
    rt = NetworkRuntime(net, state_path=tmp_path / "state.json")
    refs = [rt.submit("c", f"q{i}", "body", "criteria").ref for i in range(5)]
    for ref in refs[:4]:
        rt._questions[ref].status = "no_bounty"
    rt.save_state()
    assert list(rt._questions) == refs[2:]  # oldest finished ones dropped; the open one stays


# ---- files on a long-running host ----

def test_daily_trace_rotates_expires_and_recovers(tmp_path):
    day = 86400
    now = [1_700_000_000.0]
    legacy = tmp_path / "bridge-20231101-101010.jsonl"  # one file per restart, from older versions
    legacy.write_text("{}\n")
    os.utime(legacy, (now[0] - 20 * day, now[0] - 20 * day))
    other = tmp_path / "bridge_state.json"
    other.write_text("{}")
    os.utime(other, (now[0] - 20 * day, now[0] - 20 * day))
    trace = DailyTrace(tmp_path, "bridge", keep_days=14, clock=lambda: now[0])
    trace.write('{"a": 1}\n')
    assert not legacy.exists() and other.exists()
    first = trace.path
    now[0] += day
    trace.write('{"a": 2}\n')
    assert trace.path != first and first.read_text() == '{"a": 1}\n'

    def full(text):
        raise OSError(errno.ENOSPC, "No space left on device")

    trace._file.write = full
    try:
        trace.write('{"lost": true}\n')
        raise AssertionError("expected OSError")
    except OSError:
        pass
    trace.write('{"a": 3}\n')  # reopened by itself
    trace.close()
    assert trace.path.read_text() == '{"a": 2}\n{"a": 3}\n'
    os.utime(first, (now[0] - 15 * day, now[0] - 15 * day))
    assert prune_files(tmp_path, "bridge-*.jsonl", 14, now[0]) == [first]


def test_a_full_disk_does_not_stop_the_network(tmp_path, capsys):
    class FullDisk:
        def write(self, text):
            raise OSError(errno.ENOSPC, "No space left on device")

    net = build_network("scripted", trace=FullDisk())
    ask(net)
    net.advance(5)
    assert net.board.bounties and net.trace_errors > 5
    assert "No space left" in capsys.readouterr().err

    rt = NetworkRuntime(net, block_seconds=0.01, state_path=tmp_path / "state.json")
    rt.save_state = lambda: (_ for _ in ()).throw(OSError(errno.ENOSPC, "No space left on device"))
    height = net.board.height
    rt.start()
    deadline = time.time() + 10
    while time.time() < deadline and net.board.height < height + 5:
        time.sleep(0.02)
    rt._stop.set()  # (stop() would save once more)
    rt._thread.join(5)
    assert net.board.height >= height + 5 and rt.errors == []  # no backoff, blocks kept coming
    assert any(e["event"] == "state_save_failed" for e in net.events)
