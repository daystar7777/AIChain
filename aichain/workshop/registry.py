"""Tool workshop registry (design doc 05, 2.2), with the versioning model borrowed from
DeepSeek Harness dynamic packages: a tool holds immutable versions; `current` points at
the last version that passed and was activated, `next` at the one being tried. A failed
update never moves `current`, and rollback is just activating `current` again.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aichain.common import canonical_json, sha256_hex
from aichain.workshop import static_check
from aichain.workshop.executor import (
    DisabledExecutor,
    ExecutionDisabled,
    ExecutionFailed,
    Executor,
)
from aichain.workshop.manifest import ToolTest, ToolManifest

MIN_TESTS = 3
# Reading a published version (tool_source): capped so one read fits in an episode's context.
SOURCE_MAX_CODE_CHARS = 16_000
SOURCE_MAX_META_CHARS = 6_000


class WorkshopError(Exception):
    pass


@dataclass
class ToolVersion:
    tool_id: str
    version: int
    manifest: dict[str, Any]
    code: str
    tests: list[dict[str, Any]]
    reason: str
    code_hash: str
    created_at: float
    static_findings: list[str] = field(default_factory=list)
    test_status: str = "untested"  # untested | passed | failed | not-run (execution disabled)
    test_report: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolState:
    tool_id: str
    owner: str
    versions: dict[int, ToolVersion] = field(default_factory=dict)
    current: int | None = None
    next: int | None = None
    active: bool = False
    withdrawn: bool = False


class Workshop:
    """One workshop per network; tools are owned by the node that defined them."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS tool_versions (
        tool_id TEXT NOT NULL, version INTEGER NOT NULL, owner TEXT NOT NULL, body TEXT NOT NULL,
        PRIMARY KEY (tool_id, version));
    CREATE TABLE IF NOT EXISTS tool_state (
        tool_id TEXT PRIMARY KEY, owner TEXT NOT NULL, current INTEGER, next INTEGER,
        active INTEGER NOT NULL, withdrawn INTEGER NOT NULL);
    """

    def __init__(self, executor: Executor | None = None, path: Path | None = None) -> None:
        self.executor = executor or DisabledExecutor()
        self.tools: dict[str, ToolState] = {}
        self._lock = threading.Lock()
        self._conn = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.executescript(self.SCHEMA)
            self._load()

    @property
    def can_execute(self) -> bool:
        return not isinstance(self.executor, DisabledExecutor)

    # ---- persistence ----

    def _load(self) -> None:
        for tool_id, version, owner, body in self._conn.execute(
                "SELECT tool_id, version, owner, body FROM tool_versions ORDER BY tool_id, version"):
            st = self.tools.setdefault(tool_id, ToolState(tool_id, owner))
            st.versions[version] = ToolVersion(**json.loads(body))
        for tool_id, owner, current, nxt, active, withdrawn in self._conn.execute(
                "SELECT tool_id, owner, current, next, active, withdrawn FROM tool_state"):
            st = self.tools.setdefault(tool_id, ToolState(tool_id, owner))
            st.current, st.next, st.active, st.withdrawn = current, nxt, bool(active), bool(withdrawn)

    def _save(self, st: ToolState, version: ToolVersion | None = None) -> None:
        if self._conn is None:
            return
        with self._lock:
            if version is not None:
                self._conn.execute("INSERT OR REPLACE INTO tool_versions VALUES (?, ?, ?, ?)",
                                   (version.tool_id, version.version, st.owner,
                                    json.dumps(version.__dict__, ensure_ascii=False)))
            self._conn.execute("INSERT OR REPLACE INTO tool_state VALUES (?, ?, ?, ?, ?, ?)",
                               (st.tool_id, st.owner, st.current, st.next, int(st.active),
                                int(st.withdrawn)))
            self._conn.commit()

    # ---- lifecycle ----

    def define(self, owner: str, manifest: ToolManifest, code: str, tests: list[ToolTest],
               reason: str) -> ToolVersion:
        """Record a new immutable version: static checks, then tests (if execution is on)."""
        tool_id = f"{owner}.{manifest.name}"
        st = self.tools.get(tool_id)
        if st is not None and st.owner != owner:
            raise WorkshopError(f"{tool_id} belongs to {st.owner}")
        if len(tests) < MIN_TESTS:
            raise WorkshopError(f"at least {MIN_TESTS} test cases are required (05 doc 2.1)")
        findings = [str(f) for f in static_check.check(code)]
        if findings:
            raise WorkshopError("static check failed:\n" + "\n".join(findings))
        st = st or ToolState(tool_id, owner)
        version = max(st.versions, default=0) + 1
        tv = ToolVersion(tool_id, version, manifest.model_dump(), code,
                         [t.model_dump() for t in tests], reason,
                         sha256_hex(canonical_json({"code": code, "manifest": manifest.model_dump()})),
                         time.time())
        tv.test_status, tv.test_report = self._test(tv)
        st.versions[version] = tv
        self.tools[tool_id] = st
        self._save(st, tv)
        return tv

    def _test(self, tv: ToolVersion) -> tuple[str, list[dict[str, Any]]]:
        limits = tv.manifest["limits"]
        report = []
        for i, case in enumerate(tv.tests):
            try:
                res = self.executor.run(tv.code, case["input"], limits["timeout_s"], limits["memory_mb"])
            except ExecutionDisabled:
                return "not-run", [{"note": "execution disabled on this deployment"}]
            except ExecutionFailed as e:
                report.append({"case": i, "ok": False, "error": str(e)})
                continue
            missing = {k: v for k, v in case["expected"].items() if res.output.get(k) != v}
            report.append({"case": i, "ok": not missing, "output": res.output,
                           **({"mismatch": missing} if missing else {})})
        return ("passed" if all(r["ok"] for r in report) else "failed"), report

    def activate(self, owner: str, tool_id: str, version: int, mode: str) -> ToolState:
        """mode run: (re)start `version` (also rollback to current); update: switch versions."""
        st = self._owned(owner, tool_id)
        tv = st.versions.get(version)
        if tv is None:
            raise WorkshopError(f"{tool_id} has no version {version}")
        if mode == "update" and st.current == version:
            raise WorkshopError("already current; use mode run to restart or roll back")
        if mode == "run" and st.current not in (None, version):
            raise WorkshopError(f"current is v{st.current}; use mode update to switch versions")
        st.next = version
        if tv.test_status != "passed":
            st.active = False  # a failed update never moves current
            self._save(st)
            raise WorkshopError(f"v{version} tests are '{tv.test_status}'; only passed versions activate")
        st.current, st.next, st.active = version, None, True
        self._save(st)
        return st

    def stop(self, owner: str, tool_id: str) -> ToolState:
        st = self._owned(owner, tool_id)
        st.active = False
        self._save(st)
        return st

    def run(self, caller: str, tool_id: str, input: dict[str, Any],
            version: int | None = None) -> dict[str, Any]:
        """Run the owner's active version, or (for a licensee) the licensed `version`."""
        st = self.tools.get(tool_id)
        if st is None:
            raise WorkshopError(f"unknown tool {tool_id}")
        if st.owner == caller:
            if not st.active or st.current is None:
                raise WorkshopError(f"{tool_id} is not active")
            tv = st.versions[st.current]
        else:
            if version is None or version not in st.versions:
                raise WorkshopError("using another node's tool needs a license for a published version")
            tv = st.versions[version]
            if tv.test_status != "passed":
                raise WorkshopError(f"{tool_id} v{version} has not passed its tests")
        limits = tv.manifest["limits"]
        try:
            return self.executor.run(tv.code, input, limits["timeout_s"], limits["memory_mb"]).output
        except ExecutionDisabled as e:
            raise WorkshopError(str(e)) from e
        except ExecutionFailed as e:
            raise WorkshopError(f"tool failed: {e}") from e

    def run_tests(self, tool_id: str, version: int, tests: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
        """Run someone else's acceptance tests against a stored version (tool buy orders)."""
        st = self.tools.get(tool_id)
        if st is None or version not in st.versions:
            raise WorkshopError(f"{tool_id} v{version} is not in the workshop")
        tv = st.versions[version]
        probe = ToolVersion(tool_id, version, tv.manifest, tv.code, tests, "acceptance", tv.code_hash, 0)
        status, report = self._test(probe)
        if status == "not-run":
            raise WorkshopError("tool execution is disabled; cannot verify")
        return status == "passed", report

    def _owned(self, owner: str, tool_id: str) -> ToolState:
        st = self.tools.get(tool_id)
        if st is None:
            raise WorkshopError(f"unknown tool {tool_id}")
        if st.owner != owner:
            raise WorkshopError(f"{tool_id} belongs to {st.owner}")
        return st

    def published_source(self, tool_id: str, version: int, chain_code_hash: str) -> str:
        """Manifest, tests and code of a version for any node to review (05 doc 4.1: published
        code is public; 4.3: Warden reads it). The caller checks the version is listed on chain;
        the stored code is re-hashed against the listing and a mismatch is shown, not hidden."""
        st = self.tools.get(tool_id)
        tv = st.versions.get(version) if st is not None else None
        if tv is None:
            raise WorkshopError(f"{tool_id} v{version}: code is not in the workshop")
        actual = sha256_hex(canonical_json({"code": tv.code, "manifest": tv.manifest}))
        meta = (json.dumps(tv.manifest, ensure_ascii=False, indent=1) + "\n## tests (first 5 of "
                f"{len(tv.tests)}, status {tv.test_status})\n" + json.dumps(tv.tests[:5], ensure_ascii=False))
        if len(meta) > SOURCE_MAX_META_CHARS:
            meta = meta[:SOURCE_MAX_META_CHARS] + "\n... (cut)"
        code = tv.code
        if len(code) > SOURCE_MAX_CODE_CHARS:
            code = code[:SOURCE_MAX_CODE_CHARS] + f"\n# ... (cut: {len(tv.code)} chars in total)"
        return (f"tool_id: {tool_id}\nversion: {version}\nauthor: {st.owner}\n"
                f"code_hash: {actual} ({'matches' if actual == chain_code_hash else 'DOES NOT MATCH'} "
                f"the on-chain listing {chain_code_hash})\n## manifest\n{meta}\n"
                f"## code ({len(tv.code)} chars)\n{code}")

    def describe(self, owner: str, tool_id: str | None = None, version: int | None = None) -> str:
        if tool_id is None:
            mine = [s for s in self.tools.values() if s.owner == owner]
            if not mine:
                return "(만든 도구 없음)"
            return "\n".join(
                f"- {s.tool_id}: versions {sorted(s.versions)}, current={s.current}, next={s.next}, "
                f"active={s.active}, " + ", ".join(f"v{v.version}:{v.test_status}" for v in s.versions.values())
                for s in mine)
        st = self._owned(owner, tool_id)
        tv = st.versions.get(version or st.current or max(st.versions))
        return json.dumps({"tool_id": tv.tool_id, "version": tv.version, "manifest": tv.manifest,
                           "test_status": tv.test_status, "test_report": tv.test_report,
                           "code": tv.code}, ensure_ascii=False, indent=1)
