"""Static checks on model-written tool code (design doc 05, 2.2 step 1).

A first filter only: static checks can be bypassed, so the real boundary is the
executor's isolation. Every finding carries a line number and a hint, so the model
can fix the code in the same episode (errors that teach the fix).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

MAX_CODE_BYTES = 200 * 1024

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "ctypes", "importlib", "shutil", "pathlib", "signal",
    "multiprocessing", "threading", "asyncio", "http", "urllib", "requests", "httpx", "ftplib",
    "smtplib", "telnetlib", "pickle", "marshal", "shelve", "builtins", "inspect", "gc", "resource",
    "pty", "fcntl", "mmap", "tempfile", "glob", "io", "code", "codeop", "runpy", "site", "sqlite3",
}
ALLOWED_MODULES = {
    "json", "math", "re", "statistics", "datetime", "decimal", "fractions", "itertools",
    "functools", "collections", "string", "textwrap", "unicodedata", "hashlib", "base64",
    "random", "heapq", "bisect", "dataclasses", "typing", "enum", "operator", "copy",
    "difflib", "csv", "ast", "html", "calendar", "textwrap", "zlib", "binascii",
    # 05 doc allowlist packages (import names)
    "jsonschema", "numpy", "pandas", "sympy", "bs4", "yaml", "dateutil", "regex",
}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint", "memoryview", "help", "exit",
                   "quit"}


@dataclass
class Finding:
    line: int
    message: str
    hint: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message} — {self.hint}"


def check(code: str) -> list[Finding]:
    if len(code.encode()) > MAX_CODE_BYTES:
        return [Finding(0, "code is larger than 200KB", "split the logic or simplify")]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Finding(e.lineno or 0, f"syntax error: {e.msg}", "fix the Python syntax")]
    findings: list[Finding] = []
    has_run = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_MODULES or root not in ALLOWED_MODULES:
                    findings.append(Finding(node.lineno, f"import of '{name}' is not allowed",
                                            f"use only: {', '.join(sorted(ALLOWED_MODULES))}"))
            if isinstance(node, ast.ImportFrom) and node.level:
                findings.append(Finding(node.lineno, "relative import", "tools are single files"))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            findings.append(Finding(node.lineno, f"'{node.id}' is not allowed",
                                    "tools get their input as the `input` dict and return a dict"))
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            findings.append(Finding(node.lineno, f"dunder attribute '{node.attr}' is not allowed",
                                    "do not reach into object internals"))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            findings.append(Finding(node.lineno, "global/nonlocal is not allowed", "keep state local"))
        elif isinstance(node, ast.FunctionDef) and node.name == "run" and node in tree.body:
            args = [a.arg for a in node.args.args]
            if args[:2] != ["input", "ctx"]:
                findings.append(Finding(node.lineno, "run must be `def run(input, ctx)`",
                                        "rename the parameters to (input, ctx)"))
            has_run = True
    if not has_run:
        findings.append(Finding(1, "no top-level `def run(input, ctx)`",
                                "define the entry point: def run(input: dict, ctx) -> dict"))
    return findings
