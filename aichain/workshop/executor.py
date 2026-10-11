"""Executing model-written tool code.

`DisabledExecutor` is the default everywhere: code can be defined, checked and
versioned, but never runs. Turning execution on is an operator decision that needs a
real isolation boundary (see docs/notes/tool-execution-isolation.md); none ships enabled.

`LocalDevExecutor` exists only for development on a workstation and refuses to start
unless AICHAIN_UNSAFE_LOCAL_EXEC=1. It is *not* a security boundary: it runs the code
in a separate `python -I` process with a timeout and (where the OS allows) memory/CPU
rlimits, but it does not cut the network or the file system.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Protocol


class ExecutionDisabled(Exception):
    pass


class ExecutionFailed(Exception):
    pass


@dataclass
class ExecResult:
    output: dict[str, Any]
    elapsed_ms: int


class Executor(Protocol):
    name: str

    def run(self, code: str, input: dict[str, Any], timeout_s: int, memory_mb: int) -> ExecResult: ...


class DisabledExecutor:
    name = "disabled"

    def run(self, code: str, input: dict[str, Any], timeout_s: int, memory_mb: int) -> ExecResult:
        raise ExecutionDisabled(
            "tool execution is disabled on this deployment; the operator must enable an isolated executor")


_RUNNER = textwrap.dedent("""
    import json, sys
    def _limit(mem_mb, cpu_s):
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            if sys.platform.startswith("linux"):
                b = mem_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
        except Exception:
            pass
    req = json.loads(sys.stdin.read())
    _limit(req["memory_mb"], req["timeout_s"])
    scope = {"__name__": "aichain_tool"}
    exec(compile(req["code"], "<tool>", "exec"), scope)
    class Ctx:
        tool_id = req["tool_id"]
    out = scope["run"](req["input"], Ctx())
    if not isinstance(out, dict):
        raise TypeError("run() must return a dict")
    sys.stdout.write(json.dumps({"ok": out}))
""")


class LocalDevExecutor:
    name = "local-dev (unsafe)"

    def __init__(self) -> None:
        if os.environ.get("AICHAIN_UNSAFE_LOCAL_EXEC") != "1":
            raise ExecutionDisabled("LocalDevExecutor needs AICHAIN_UNSAFE_LOCAL_EXEC=1 (development only)")

    def run(self, code: str, input: dict[str, Any], timeout_s: int, memory_mb: int,
            tool_id: str = "dev") -> ExecResult:
        import time

        started = time.monotonic()
        req = json.dumps({"code": code, "input": input, "timeout_s": timeout_s,
                          "memory_mb": memory_mb, "tool_id": tool_id})
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=req, text=True,
                                  capture_output=True, timeout=timeout_s + 2,
                                  env={"PATH": "/usr/bin:/bin"}, cwd="/")
        except subprocess.TimeoutExpired as e:
            raise ExecutionFailed(f"timed out after {timeout_s}s") from e
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            raise ExecutionFailed(err[-1] if err else f"exit code {proc.returncode}")
        try:
            out = json.loads(proc.stdout)["ok"]
        except (ValueError, KeyError) as e:
            raise ExecutionFailed("tool produced no JSON result") from e
        return ExecResult(out, int((time.monotonic() - started) * 1000))


# Each call runs in a fresh transient unit created by systemd (as root, so no user namespaces
# are needed): throwaway user, no network at all, read-only system, no access to AIChain data,
# capped memory/tasks/time. See docs/notes/tool-execution-isolation.md (option A).
SYSTEMD_PROPERTIES = [
    "DynamicUser=yes",
    "PrivateNetwork=yes",
    "IPAddressDeny=any",
    "RestrictAddressFamilies=none",
    "ProtectSystem=strict",
    "ProtectHome=yes",
    "PrivateTmp=yes",
    "PrivateDevices=yes",
    "PrivateUsers=yes",
    "NoNewPrivileges=yes",
    "CapabilityBoundingSet=",
    "AmbientCapabilities=",
    "SystemCallFilter=@system-service",
    "SystemCallFilter=~@privileged @mount @debug @module @raw-io @reboot @swap @cpu-emulation @obsolete",
    "SystemCallArchitectures=native",
    "ProtectKernelTunables=yes",
    "ProtectKernelModules=yes",
    "ProtectKernelLogs=yes",
    "ProtectControlGroups=yes",
    "ProtectClock=yes",
    "ProtectHostname=yes",
    "ProtectProc=invisible",
    "ProcSubset=pid",
    "RestrictNamespaces=yes",
    "RestrictRealtime=yes",
    "RestrictSUIDSGID=yes",
    "LockPersonality=yes",
    "MemoryDenyWriteExecute=yes",
    "UMask=0077",
    "TasksMax=64",
    "LimitNOFILE=64",
    "CPUQuota=100%",
    "MemorySwapMax=0",
    "InaccessiblePaths=-/opt/aichain -/opt/aichain-browser -/root -/home -/var/lib/aichain-browser "
    "-/run/aichain-egress -/run/aichain-browser -/etc/ssh",
]


class SystemdExecutor:
    name = "systemd-transient"

    def __init__(self, python: str = "/usr/bin/python3", systemd_run: str = "/usr/bin/systemd-run") -> None:
        if not sys.platform.startswith("linux") or os.geteuid() != 0:
            raise ExecutionDisabled("SystemdExecutor needs Linux and root (it creates transient units)")
        if not os.path.exists(systemd_run):
            raise ExecutionDisabled(f"{systemd_run} not found")
        self.python = python
        self.systemd_run = systemd_run

    def command(self, timeout_s: int, memory_mb: int) -> list[str]:
        props = SYSTEMD_PROPERTIES + [f"MemoryMax={memory_mb}M", f"RuntimeMaxSec={timeout_s}"]
        cmd = [self.systemd_run, "--wait", "--pipe", "--collect", "--quiet", "--service-type=exec"]
        for p in props:
            cmd += ["-p", p]
        return cmd + ["--", self.python, "-I", "-S", "-c", _RUNNER]

    def run(self, code: str, input: dict[str, Any], timeout_s: int, memory_mb: int,
            tool_id: str = "tool") -> ExecResult:
        import time

        started = time.monotonic()
        req = json.dumps({"code": code, "input": input, "timeout_s": timeout_s,
                          "memory_mb": memory_mb, "tool_id": tool_id})
        try:
            proc = subprocess.run(self.command(timeout_s, memory_mb), input=req, text=True,
                                  capture_output=True, timeout=timeout_s + 10)
        except subprocess.TimeoutExpired as e:
            raise ExecutionFailed(f"timed out after {timeout_s}s") from e
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            raise ExecutionFailed(err[-1][:300] if err else f"exit code {proc.returncode}")
        try:
            out = json.loads(proc.stdout)["ok"]
        except (ValueError, KeyError) as e:
            raise ExecutionFailed("tool produced no JSON result") from e
        return ExecResult(out, int((time.monotonic() - started) * 1000))
