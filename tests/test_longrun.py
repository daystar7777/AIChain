"""Running for months: bounded memory, rotating files, a full disk, halts and the watchdog."""

import errno
import json
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
    rt._wake.set()
    for t in rt._threads:
        t.join(5)
    assert net.board.height >= height + 5 and rt.errors == []  # no backoff, blocks kept coming
    assert any(e["event"] == "state_save_failed" for e in net.events)


# ---- halts, watchdog, alerts ----

def test_a_halted_chain_alerts_then_asks_the_process_to_exit():
    net = build_network("scripted")
    rt = NetworkRuntime(net, block_seconds=0.01)
    rt.halt_exit_seconds = 0.3
    reasons = []
    rt.on_fatal = reasons.append
    rt.start()
    deadline = time.time() + 5
    while time.time() < deadline and net.board.height < 2:
        time.sleep(0.01)
    net.board.halted = "chain log write failed: disk full"
    halted_at = time.time()
    while time.time() < deadline and not reasons:
        time.sleep(0.02)
    rt._stop.set()
    rt._wake.set()
    for t in rt._threads:
        t.join(5)
    assert reasons == ["chain log write failed: disk full"] and time.time() - halted_at >= 0.3
    assert "체인이 멈췄어요" in net.manager_outbox[-1]["text"] and "systemd" in net.manager_outbox[-1]["text"]
    assert [e["reason"] for e in net.events if e["event"] == "halted"] == [net.board.halted]


def test_runtime_progress_follows_log_entries_and_llm_calls():
    net = build_network("scripted")
    rt = NetworkRuntime(net)
    assert rt.progress_at == 0.0
    before = time.monotonic()
    net.log("core", "x")
    assert rt.progress_at >= before
    rt.progress_at = 0.0
    ask(net)  # inside an episode: LLM calls and tool entries count as progress
    assert rt.progress_at >= before


def notify_socket(monkeypatch):
    import socket
    import uuid

    name = f"aichain-test-{uuid.uuid4().hex}"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind("\0" + name)
    sock.settimeout(2)
    monkeypatch.setenv("NOTIFY_SOCKET", "@" + name)
    return sock


def test_watchdog_reports_ready_pings_while_alive_and_stops_when_stalled(monkeypatch, capsys):
    from aichain.ops import Watchdog, sd_notify

    sock = notify_socket(monkeypatch)
    monkeypatch.setenv("WATCHDOG_USEC", "600000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid()))
    progress = [0.0]
    wd = Watchdog(lambda: progress[0], stall_seconds=60, status=lambda: "block #7")
    assert wd.enabled and abs(wd.interval - 0.2) < 1e-9
    assert wd.beat(time.monotonic()) is False  # still starting: nothing sent
    progress[0] = time.monotonic()
    assert wd.beat(time.monotonic()) is True
    assert sock.recv(100) == b"READY=1"
    assert sock.recv(100) == b"WATCHDOG=1\nSTATUS=block #7"
    assert wd.beat(time.monotonic() + 61) is False and wd.stalled  # no progress for 61 s: no ping
    assert "no more watchdog pings" in capsys.readouterr().err
    wd.start()  # the thread pings by itself
    progress[0] = time.monotonic() + 3600
    assert sock.recv(100).startswith(b"WATCHDOG=1")
    wd.stop()
    monkeypatch.delenv("NOTIFY_SOCKET")
    assert sd_notify("READY=1") is False  # not under systemd: a no-op
    monkeypatch.setenv("WATCHDOG_PID", "1")
    assert not Watchdog(lambda: 1.0).enabled  # the watchdog belongs to another process


def run_alert(tmp_path, env_text, sent, fail=None, unit="aichain-bridge.service"):
    from aichain import alert

    env = tmp_path / ".env"
    env.write_text(env_text)

    def run(cmd):
        if cmd[0] == "systemctl":
            return ("Result=exit-code\nExecMainCode=1\nExecMainStatus=4\nNRestarts=3\nRestart=on-failure\n"
                    "ActiveState=activating\n")
        return "chain halted: chain log write failed: disk full\nkey sk-" + "A1b2C3d4" * 6

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def urlopen(req, timeout):
        if fail:
            raise fail
        sent.append((req.full_url, json.loads(req.data)))
        return Response()

    return alert.main(["--unit", unit, "--env-file", str(env), "--state", str(tmp_path / "state.json")],
                      run=run, urlopen=urlopen)


def test_failure_alert_goes_to_telegram_once_per_window(tmp_path, capsys):
    token = "123456789:" + "T" * 35
    env = f"TELEGRAM_BOT_TOKEN={token}\nTELEGRAM_MANAGER_CHAT_ID=42\n"
    sent = []
    assert run_alert(tmp_path, env, sent) == 0
    (url, body), = sent
    assert url == f"https://api.telegram.org/bot{token}/sendMessage" and body["chat_id"] == "42"
    text = body["text"]
    assert "aichain-bridge.service 실패" in text and "종료 코드 4" in text and "재시작 3회째" in text
    assert "disk full" in text and "sk-A1b2" not in text  # log lines are masked
    assert run_alert(tmp_path, env, sent) == 0 and len(sent) == 1  # crash loop: quiet for 30 min
    state = json.loads((tmp_path / "state.json").read_text())
    state["aichain-bridge.service"]["last"] -= 31 * 60
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert run_alert(tmp_path, env, sent) == 0 and "1번 더 실패" in sent[-1][1]["text"]
    out = capsys.readouterr()
    assert token not in out.out + out.err


def test_failure_alert_never_prints_the_token(tmp_path, capsys):
    import urllib.error

    token = "123456789:" + "T" * 35
    env = f"TELEGRAM_BOT_TOKEN={token}\nTELEGRAM_MANAGER_CHAT_ID=42\n"
    fail = urllib.error.URLError(f"cannot reach https://api.telegram.org/bot{token}/sendMessage")
    assert run_alert(tmp_path, env, [], fail=fail) == 1
    out = capsys.readouterr()
    assert token not in out.out + out.err and "***" in out.err
    unset = tmp_path / "unset"
    unset.mkdir()
    assert run_alert(unset, "", []) == 0  # no token: a line in the journal, not a failure
    assert "no alert sent" in capsys.readouterr().out


def test_bridge_process_exits_with_code_4_after_a_halt(tmp_path, monkeypatch):
    import socket
    import threading

    from aichain.bridge import __main__ as bridge_main
    from aichain.bridge.tokens import add_client

    monkeypatch.setattr(bridge_main, "ROOT", tmp_path)  # data/ under tmp_path
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    tokens = tmp_path / "tokens.json"
    add_client(tokens, "c", 5)
    runtimes = []

    class Recording(NetworkRuntime):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            runtimes.append(self)

    monkeypatch.setattr(bridge_main, "NetworkRuntime", Recording)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    def halt_once_running():
        deadline = time.time() + 30
        while time.time() < deadline and not (runtimes and runtimes[0].progress_at):
            time.sleep(0.05)
        runtimes[0].net.board.halted = "chain log write failed: test"

    threading.Thread(target=halt_once_running, daemon=True).start()
    code = bridge_main.main(["--tokens", str(tokens), "--port", str(port), "--block-seconds", "0.05",
                             "--halt-exit-seconds", "0.3"])
    assert code == bridge_main.EXIT_HALTED == 4
    assert list((tmp_path / "data").glob("bridge-*.jsonl"))  # the daily trace file
    assert "체인이 멈췄어요" in runtimes[0].net.manager_outbox[-1]["text"]
