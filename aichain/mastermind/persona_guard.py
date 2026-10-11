"""persona_edit checks, run by Core in order (design doc 04, 5.3). The LLM proposes; this
code decides. Every verdict, approved or not, is recorded as a persona memory entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 2000
MAX_CHANGE_RATIO = 0.30
COOLDOWN_BLOCKS = 17_280  # 24 hours

# Check 6: patterns a self-edit must never contain.
_RULES = r"(규칙|코어|지시|원칙|제약|rule|core|instruction|guard|policy)"
_DEFY = r"(무시|어기|어겨|따르지\s*않|지키지\s*않|ignore|disregard|override|bypass)"
_ACCOUNT = r"(?<![A-Za-z0-9])(n\d+|operator|root)(?![A-Za-z0-9])"  # \b fails before Hangul

FORBIDDEN = [
    (re.compile(_DEFY + r".{0,15}" + _RULES, re.I | re.S), "attempt to ignore or override rules"),
    (re.compile(_RULES + r".{0,15}" + _DEFY, re.I | re.S), "attempt to ignore or override rules"),
    (re.compile(r"(고정\s*코어|fixed\s*core|system\s*prompt|시스템\s*프롬프트).{0,20}"
                r"(보다|우선|대신|over|instead|replace|바꾸|변경)", re.I | re.S),
     "attempt to change the priority of the fixed core"),
    (re.compile(r"(우선\s*순위|priority).{0,15}(바꾸|변경|높이|올리|change|raise)", re.I | re.S),
     "attempt to change rule priority"),
    (re.compile(r"(개인\s*키|비밀\s*키|private\s*key|api[\s_-]*key|키\s*파일|설정\s*파일|config|\.env|"
                r"policy\s*guard|지출\s*한도|송금\s*한도|한도를|spend(ing)?\s*limit|예산\s*상한)", re.I),
     "mentions keys, configuration or limits"),
    (re.compile(r"<\s*/?\s*data\b|data\s*(블록|태그|block|tag)", re.I),
     "manipulates <data> blocks"),
    (re.compile(_ACCOUNT + r".{0,25}(무조건|항상|언제나|always|unconditionally).{0,25}"
                r"(믿|신뢰|따르|송금|보내|trust|follow|transfer|send)", re.I | re.S),
     "unconditional trust or transfers toward a specific account"),
    (re.compile(r"(무조건|항상|always|unconditionally).{0,15}(송금|이체|transfer|send\s+points)", re.I | re.S),
     "unconditional transfers"),
]


@dataclass
class Verdict:
    approved: bool
    failed: list[int]
    reasons: list[str]


def edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, stopping early once it exceeds `limit` (returns limit + 1)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            best = min(best, cur[j])
        if best > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def check(new_text: str, old_text: str, *, in_reflect: bool, height: int, last_edit_height: int | None,
          refs_sources: list[str]) -> Verdict:
    failed: list[int] = []
    reasons: list[str] = []

    def fail(n: int, why: str) -> None:
        failed.append(n)
        reasons.append(f"#{n} {why}")

    if not in_reflect:
        fail(1, "persona_edit is only allowed in a reflect episode")
    if last_edit_height is not None and height - last_edit_height < COOLDOWN_BLOCKS:
        fail(2, f"last edit was {height - last_edit_height} blocks ago (24h = {COOLDOWN_BLOCKS})")
    if len(new_text) > MAX_CHARS:
        fail(3, f"{len(new_text)} chars > {MAX_CHARS}")
    limit = int(len(old_text) * MAX_CHANGE_RATIO)
    if not failed or 3 not in failed:
        dist = edit_distance(old_text, new_text, limit)
        if dist > limit:
            fail(4, f"changes more than {int(MAX_CHANGE_RATIO * 100)}% of the text (limit {limit} chars)")
    if not any(s == "self" or s.startswith("chain:") for s in refs_sources):
        fail(5, "refs must include at least one memory from your own experience (self or chain:)")
    for pattern, why in FORBIDDEN:
        if pattern.search(new_text):
            fail(6, why)
            break
    return Verdict(not failed, failed, reasons)
