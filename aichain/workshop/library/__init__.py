"""Standard tool library: installs and publishes the catalog under the `lib` account."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aichain.board import BoardError
from aichain.workshop.manifest import ToolManifest, ToolTest
from aichain.workshop.registry import Workshop, WorkshopError

from .catalog import CATALOG

LIB = "lib"
TOOLS_DIR = Path(__file__).parent / "tools"
_JSON_TYPES = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean",
               "array": "array", "object": "object"}


def manifest_for(entry: dict[str, Any]) -> ToolManifest:
    props = {k: ({"type": _JSON_TYPES[v]} if v in _JSON_TYPES else {}) for k, v in entry["input"].items()}
    return ToolManifest(name=entry["name"], description=entry["description"],
                        input_schema={"type": "object", "properties": props},
                        output_schema={"type": "object"})


def install(workshop: Workshop, board: Any) -> dict[str, Any]:
    """Define (or re-define on change), test, activate and publish every library tool.
    Idempotent: unchanged tools are left as they are. Tools that fail their tests are skipped."""
    if LIB not in board.accounts:
        board.open_account(LIB, 0)
    report: dict[str, Any] = {"installed": [], "unchanged": [], "failed": []}
    for entry in CATALOG:
        code = (TOOLS_DIR / entry["file"]).read_text(encoding="utf-8")
        manifest = manifest_for(entry)
        tool_id = f"{LIB}.{entry['name']}"
        st = workshop.tools.get(tool_id)
        current = st.versions.get(st.current) if st and st.current else None
        if current and current.code == code and current.manifest == manifest.model_dump():
            if not any(t.tool_id == tool_id and t.version == current.version for t in board.tools.values()):
                board.publish_tool(LIB, tool_id, current.version, current.code_hash, 0, 0)
            report["unchanged"].append(tool_id)
            continue
        try:
            tv = workshop.define(LIB, manifest, code, [ToolTest(**x) for x in entry["tests"]],
                                 "standard library")
            if tv.test_status != "passed":
                report["failed"].append({"tool": tool_id, "status": tv.test_status, "report": tv.test_report})
                continue
            workshop.activate(LIB, tool_id, tv.version, "update" if current else "run")
            board.publish_tool(LIB, tool_id, tv.version, tv.code_hash, 0, 0)
            report["installed"].append(f"{tool_id}@{tv.version}")
        except (WorkshopError, BoardError) as e:
            report["failed"].append({"tool": tool_id, "error": str(e)[:300]})
    return report


def summary() -> str:
    return "\n".join(f"- lib.{e['name']}: {e['description']}" for e in CATALOG)
