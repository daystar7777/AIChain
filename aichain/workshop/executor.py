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
