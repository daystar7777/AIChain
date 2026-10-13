import json
import os
import sys

import pytest

from aichain.gateways.llm.scripted import tool_response
from aichain.sim.factory import build_network
from aichain.workshop import static_check
from aichain.workshop.executor import DisabledExecutor, ExecutionDisabled, LocalDevExecutor
from aichain.workshop.manifest import Limits, ToolManifest, ToolTest
from aichain.workshop.registry import Workshop, WorkshopError

MANIFEST = ToolManifest(
    name="word-count", description="텍스트의 단어 수를 센다",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"words": {"type": "integer"}}})
GOOD = "def run(input, ctx):\n    return {'words': len(input['text'].split())}\n"
BUGGY = "def run(input, ctx):\n    return {'words': len(input['text'])}\n"
TESTS = [ToolTest(input={"text": "a b c"}, expected={"words": 3}),
         ToolTest(input={"text": ""}, expected={"words": 0}),
         ToolTest(input={"text": "하나 둘"}, expected={"words": 2})]


@pytest.fixture
def dev_exec(monkeypatch):
    # Executing code without isolation is for the developer workstation only, never the server.
    if not sys.platform.startswith("darwin") and os.environ.get("AICHAIN_DEV_EXEC_TESTS") != "1":
        pytest.skip("code-execution tests run only on a development Mac")
    monkeypatch.setenv("AICHAIN_UNSAFE_LOCAL_EXEC", "1")
    return LocalDevExecutor()


@pytest.mark.parametrize("code,needle", [
    ("import os\ndef run(input, ctx):\n    return {}", "import of 'os'"),
    ("import socket\ndef run(input, ctx):\n    return {}", "import of 'socket'"),
    ("from subprocess import run as r\ndef run(input, ctx):\n    return {}", "subprocess"),
    ("def run(input, ctx):\n    return eval('1')", "'eval'"),
    ("def run(input, ctx):\n    return open('/etc/passwd').read()", "'open'"),
    ("def run(input, ctx):\n    return ().__class__.__bases__", "dunder"),
    ("def go(input, ctx):\n    return {}", "no top-level"),
    ("def run(x):\n    return {}", "run must be"),
    ("def run(input, ctx)\n    return {}", "syntax error"),
])
def test_static_check_findings_teach_the_fix(code, needle):
    findings = static_check.check(code)
    assert any(needle in str(f) for f in findings), findings
    assert all(f.hint for f in findings)


def test_static_check_accepts_plain_tool():
    assert static_check.check("import json, math\n" + GOOD) == []


def test_manifest_validation():
    with pytest.raises(ValueError):
        ToolManifest(name="Bad Name", description="x" * 20, input_schema={"type": "object"},
                     output_schema={"type": "object"})
    with pytest.raises(ValueError):
        ToolManifest(name="ok-name", description="x" * 20, input_schema={"type": "object"},
                     output_schema={"type": "object"}, python_packages=["requests"])


def test_disabled_executor_is_the_default(tmp_path):
    ws = Workshop(path=tmp_path / "w.db")
    assert not ws.can_execute
    tv = ws.define("n2", MANIFEST, GOOD, TESTS, "단어 수 계산이 잦다")
    assert tv.test_status == "not-run"
    with pytest.raises(WorkshopError, match="not-run"):
        ws.activate("n2", tv.tool_id, 1, "run")
    with pytest.raises(ExecutionDisabled):
        DisabledExecutor().run(GOOD, {}, 1, 64)
    with pytest.raises(ExecutionDisabled):
        LocalDevExecutor()  # env flag not set


def test_versions_pointers_and_rollback(tmp_path, dev_exec):
    ws = Workshop(dev_exec, tmp_path / "w.db")
    v1 = ws.define("n2", MANIFEST, GOOD, TESTS, "first")
    assert v1.test_status == "passed"
    ws.activate("n2", "n2.word-count", 1, "run")
    assert ws.run("n2", "n2.word-count", {"text": "x y"}) == {"words": 2}

    v2 = ws.define("n2", MANIFEST, BUGGY, TESTS, "buggy refactor")
    assert v2.test_status == "failed" and any("mismatch" in r for r in v2.test_report)
    with pytest.raises(WorkshopError):
        ws.activate("n2", "n2.word-count", 2, "update")
    st = ws.tools["n2.word-count"]
    assert (st.current, st.next) == (1, 2)  # failed update did not move current
    with pytest.raises(WorkshopError):
        ws.activate("n2", "n2.word-count", 2, "run")  # switching needs update
    ws.activate("n2", "n2.word-count", 1, "run")  # rollback/restart current
    assert st.active and st.current == 1

    # Persistence: a new workshop sees the same versions and pointers.
    again = Workshop(dev_exec, tmp_path / "w.db")
    assert sorted(again.tools["n2.word-count"].versions) == [1, 2]
    assert again.tools["n2.word-count"].current == 1
    # Other nodes cannot touch or run it.
    with pytest.raises(WorkshopError):
        again.activate("n1", "n2.word-count", 1, "run")
    with pytest.raises(WorkshopError):
        again.run("n1", "n2.word-count", {"text": "a"})


def test_define_requires_tests_and_clean_code(tmp_path):
    ws = Workshop(path=tmp_path / "w.db")
    with pytest.raises(WorkshopError, match="test cases"):
        ws.define("n2", MANIFEST, GOOD, TESTS[:2], "r")
    with pytest.raises(WorkshopError, match="static check"):
        ws.define("n2", MANIFEST, "import os\n" + GOOD, TESTS, "r")


def test_timeouts_and_crashes_are_reported(tmp_path, dev_exec):
    ws = Workshop(dev_exec)
    loop = "def run(input, ctx):\n    while True:\n        pass\n"
    m = MANIFEST.model_copy(update={"limits": Limits(timeout_s=1, memory_mb=64)})
    tv = ws.define("n2", m, loop, TESTS, "r")
    assert tv.test_status == "failed" and "error" in tv.test_report[0]


def test_mastermind_defines_and_repairs_a_tool(tmp_path, dev_exec):
    def builder(request):
        tool_msgs = [m for m in request.messages if m.role == "tool"]
        base = {"manifest": json.loads(MANIFEST.model_dump_json()),
                "tests": [t.model_dump() for t in TESTS], "reason": "단어 수 계산"}
        if not tool_msgs:
            return tool_response(("tool_dev", {**base, "code": BUGGY}))
        if len(tool_msgs) == 1:  # saw the failing report → fix and define v2
            assert "failed" in tool_msgs[0].content
            return tool_response(("tool_dev", {**base, "code": GOOD}))
        if len(tool_msgs) == 2:
            return tool_response(("tool_activate", {"tool_id": "n2.word-count", "version": 2, "mode": "run"}))
        if len(tool_msgs) == 3:
            return tool_response(("tool_run", {"tool_id": "n2.word-count", "input": {"text": "a b"}}))
        return tool_response(("done", {"summary": "도구 v2 활성화"}))

    net = build_network("scripted", brain=builder, workshop=Workshop(dev_exec, tmp_path / "w.db"))
    n2 = net.nodes["n2"]
    assert {"tool_dev", "tool_list", "tool_activate", "tool_run"} <= set(n2.tool_names())
    n2.enqueue("idle_tick", {})
    net.run_until_idle()
    kinds = [(e["event"], e.get("ok")) for e in net.events if e["event"].startswith("tool_")]
    assert kinds == [("tool_defined", True), ("tool_defined", True), ("tool_activated", True)]
    tool_out = [e for e in net.events if e["event"] == "tool" and e["tool"] == "tool_run"]
    assert '"words": 2' in tool_out[0]["result"]
    assert "도구 공방" in net.gateway.adapters["scripted"].requests[0].messages[1].content


def test_tool_run_hidden_when_execution_disabled(tmp_path):
    net = build_network("scripted", workshop=Workshop())
    names = net.nodes["n1"].tool_names()
    assert "tool_dev" in names and "tool_run" not in names
    assert "꺼져 있다" in net.nodes["n1"]._workshop_section()
    assert "tool_dev" not in build_network("scripted").nodes["n1"].tool_names()


def test_systemd_executor_command_is_locked_down():
    from aichain.workshop.executor import SYSTEMD_PROPERTIES, SystemdExecutor

    ex = SystemdExecutor.__new__(SystemdExecutor)  # build the command without needing Linux/root
    ex.python, ex.systemd_run = "/usr/bin/python3", "/usr/bin/systemd-run"
    cmd = ex.command(7, 128)
    props = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-p"]
    for required in ("DynamicUser=yes", "PrivateNetwork=yes", "IPAddressDeny=any", "ProtectSystem=strict",
                     "NoNewPrivileges=yes", "CapabilityBoundingSet=", "TasksMax=64",
                     "MemoryMax=128M", "RuntimeMaxSec=7"):
        assert required in props, required
    assert any(p.startswith("InaccessiblePaths=") and "/opt/aichain" in p for p in props)
    assert cmd[-5:-1] == ["/usr/bin/python3", "-I", "-S", "-c"]
    assert len(props) == len(SYSTEMD_PROPERTIES) + 2
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        with pytest.raises(ExecutionDisabled):
            SystemdExecutor()


def test_tool_package_list_matches_the_allowlist():
    from pathlib import Path

    from aichain.workshop.manifest import ALLOWED_PACKAGES

    path = Path(__file__).resolve().parents[1] / "deploy" / "tool-packages.txt"
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith("#")]
    assert all("==" in x for x in lines)  # pinned
    assert {x.split("==")[0].lower() for x in lines} == ALLOWED_PACKAGES


def test_sandbox_runner_sees_only_the_packages_dir(tmp_path):
    # The runner itself (no tool isolation needed: the code here is the test's own).
    import subprocess

    from aichain.workshop.executor import _RUNNER

    pkg = tmp_path / "pkgs" / "fakepkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 7\n")
    code = ("def run(input, ctx):\n    import fakepkg, os\n    try:\n        import pydantic\n"
            "        hidden = False\n    except ImportError:\n        hidden = True\n"
            "    return {'value': fakepkg.VALUE, 'site_hidden': hidden,\n"
            "            'blas': os.environ.get('OPENBLAS_NUM_THREADS')}\n")
    req = {"code": code, "input": {}, "timeout_s": 10, "memory_mb": 256, "tool_id": "t",
           "sys_path": [str(tmp_path / "pkgs")]}
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", _RUNNER], input=json.dumps(req), text=True,
                          capture_output=True, timeout=30, cwd="/")
    assert json.loads(proc.stdout)["ok"] == {"value": 7, "site_hidden": True, "blas": "1"}, proc.stderr
    del req["sys_path"]  # without the packages dir nothing beyond the stdlib is importable
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", _RUNNER], input=json.dumps(req), text=True,
                          capture_output=True, timeout=30, cwd="/")
    assert proc.returncode != 0 and "No module named 'fakepkg'" in proc.stderr


def test_systemd_executor_passes_packages_dir_and_memory_headroom(monkeypatch, tmp_path):
    import subprocess

    from aichain.workshop import executor as ex_mod

    ex = ex_mod.SystemdExecutor.__new__(ex_mod.SystemdExecutor)
    ex.python, ex.systemd_run, ex.packages_dir = "/usr/bin/python3", "/usr/bin/systemd-run", str(tmp_path / "none")
    seen = {}

    def fake_run(cmd, input, **kw):
        seen["req"] = json.loads(input)
        if "numpy" in seen["req"]["code"]:
            return subprocess.CompletedProcess(cmd, 1, "", "ModuleNotFoundError: No module named 'numpy'")
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"ok": {"x": 1}}), "")

    monkeypatch.setattr(ex_mod.subprocess, "run", fake_run)
    assert ex.run("def run(input, ctx):\n    return {'x': 1}\n", {}, 5, 128).output == {"x": 1}
    assert seen["req"]["sys_path"] == [str(tmp_path / "none")]
    assert seen["req"]["memory_mb"] == 128
    assert seen["req"]["address_space_mb"] == 128 + ex_mod.ADDRESS_SPACE_HEADROOM_MB
    with pytest.raises(ex_mod.ExecutionFailed, match="not installed on this server"):
        ex.run("import numpy\ndef run(input, ctx):\n    return {}\n", {}, 5, 128)
    # The system's own Python packages are not reachable from the unit.
    assert any("/usr/lib/python3/dist-packages" in p for p in ex_mod.SYSTEMD_PROPERTIES)
