"""Experiment report (M7) from the journal and the knowledge store.

    python -m aichain.report                    # last 24 hours -> data/reports/report-<date>.md/.html
    python -m aichain.report --hours 336        # two weeks
    python -m aichain.report --all

Read-only: opens the databases in read-only mode.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# OpenCode Go list price for muse-spark-1.3-contributor, USD per 1M tokens (input, output).
PRICE_PER_M = (0.1, 0.2)


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


PROBLEM_KINDS = ("tool_invalid", "policy_reject", "board_reject", "tool_error", "episode_error", "web_error",
                 "runtime_error", "busy")


def collect(journal: Path, knowledge: Path, since: float | None) -> dict[str, Any]:
    """Aggregates for the report, computed by SQLite or streamed row by row: no table is loaded
    into memory, and LLM prompts (the bulk of the journal) are never read into Python. With
    `since`, only the period's rows are read (through the ts index the journal writer creates)."""
    j = _ro(journal)
    where, args = ("WHERE ts >= ?", (since,)) if since else ("", ())
    and_where = "AND ts >= ?" if since else ""
    by_kind = Counter(dict(j.execute(f"SELECT kind, COUNT(*) FROM entries {where} GROUP BY kind", args)))
    tok_in, tok_out, failed = j.execute(
        "SELECT COALESCE(SUM(json_extract(data, '$.response.usage.input_tokens')), 0), "
        "COALESCE(SUM(json_extract(data, '$.response.usage.output_tokens')), 0), "
        "COALESCE(SUM(COALESCE(json_extract(data, '$.error'), '') != ''), 0) "
        f"FROM entries WHERE kind = 'llm_call' {and_where}", args).fetchone()
    episodes: dict[str, Counter] = defaultdict(Counter)
    tools_used: dict[str, Counter] = defaultdict(Counter)
    domains: Counter = Counter()
    for node, kind, ev_kind, stopped, tool, url in j.execute(
            "SELECT node, kind, json_extract(data, '$.kind'), json_extract(data, '$.stopped'), "
            "json_extract(data, '$.tool'), json_extract(data, '$.url') FROM entries "
            f"WHERE kind IN ('episode_end', 'tool', 'web_fetch') {and_where}", args):
        if kind == "episode_end":
            episodes[node][f"{ev_kind}/{stopped}"] += 1
        elif kind == "tool":
            tools_used[node][tool] += 1
        else:
            url = url or ""
            domains[url.split("/")[2] if "//" in url else url] += 1
    first = j.execute(f"SELECT ts, height FROM entries {where} ORDER BY id LIMIT 1", args).fetchone()
    last = j.execute(f"SELECT ts, height FROM entries {where} ORDER BY id DESC LIMIT 1", args).fetchone()
    m_first = j.execute(f"SELECT data FROM metrics {where} ORDER BY ts LIMIT 1", args).fetchone()
    m_last = j.execute(f"SELECT data FROM metrics {where} ORDER BY ts DESC LIMIT 1", args).fetchone()
    runs = [dict(r) for r in j.execute("SELECT id, started_at, ended_at, label FROM runs ORDER BY id")]
    j.close()
    notes: dict[str, list] = defaultdict(list)
    persona: dict[str, list] = defaultdict(list)
    web_facts: Counter = Counter()
    if knowledge.exists():
        k = _ro(knowledge)
        cols = "node, seq, ts, height, kind, source, trust, content, topic, data, supersedes, hash"
        # Current notes only (not superseded by a newer version); every persona edit.
        for r in k.execute(f"SELECT {cols} FROM memories WHERE kind = 'note' AND hash NOT IN "
                           "(SELECT supersedes FROM memories WHERE supersedes IS NOT NULL) ORDER BY node, seq"):
            notes[r["node"]].append(dict(r) | {"data": json.loads(r["data"]) if r["data"] else {}})
        for r in k.execute(f"SELECT {cols} FROM memories WHERE kind = 'persona' AND data IS NOT NULL "
                           "AND data != '' ORDER BY node, seq"):
            persona[r["node"]].append(dict(r) | {"data": json.loads(r["data"])})
        web_facts.update(dict(k.execute(
            "SELECT node, COUNT(*) FROM memories WHERE kind = 'fact' AND substr(source, 1, 4) = 'web:' "
            "AND ts >= ? GROUP BY node", (since or 0,))))
        k.close()
    return {
        "by_kind": by_kind, "tokens": (tok_in, tok_out), "failed_calls": failed,
        "episodes": episodes, "tools_used": tools_used, "domains": domains,
        "period": (first["ts"], last["ts"]) if first else (None, None),
        "heights": (first["height"], last["height"]) if first else (None, None),
        "first": json.loads(m_first["data"]) if m_first else None,
        "last": json.loads(m_last["data"]) if m_last else None,
        "runs": runs, "notes": notes, "persona": persona, "web_facts": web_facts, "since": since,
    }


def analyse(d: dict[str, Any]) -> dict[str, Any]:
    tok_in, tok_out = d["tokens"]
    return {
        **d,
        "llm_calls": d["by_kind"].get("llm_call", 0),
        "cost_usd": tok_in / 1e6 * PRICE_PER_M[0] + tok_out / 1e6 * PRICE_PER_M[1],
        "problems": Counter({k: v for k, v in d["by_kind"].items() if k in PROBLEM_KINDS}),
    }


def _fmt_ts(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def _p(mp: int | None) -> str:
    return "-" if mp is None else f"{mp / 1000:,.1f}P"


def render_markdown(a: dict[str, Any], title: str) -> str:
    out = [f"# {title}", "",
           f"기간: {_fmt_ts(a['period'][0])} ~ {_fmt_ts(a['period'][1])} · 블록 #{a['heights'][0]} ~ #{a['heights'][1]}", ""]
    tin, tout = a["tokens"]
    out += ["## 요약", "",
            f"- LLM 호출 {a['llm_calls']}회 (실패 {a['failed_calls']}), 토큰 입력 {tin:,} / 출력 {tout:,}, "
            f"추정 비용 ${a['cost_usd']:.3f}",
            f"- 에피소드 {a['by_kind'].get('episode_end', 0)}개, 도구 호출 {a['by_kind'].get('tool', 0)}회",
            f"- 웹 검색 {a['by_kind'].get('web_search', 0)}회, 페이지 읽기 {a['by_kind'].get('web_fetch', 0)}회, "
            f"지식 노트 갱신 {a['by_kind'].get('knowledge', 0)}회",
            f"- 현상금 정산 {a['by_kind'].get('bounty_settled', 0)}건, 분쟁 {a['by_kind'].get('dispute_opened', 0)}건, "
            f"평가 {a['by_kind'].get('rated', 0)}건, 제안 {a['by_kind'].get('proposal_created', 0)}건, "
            f"도구 등록 {a['by_kind'].get('tool_published', 0)}건",
            f"- 성향 수정 시도 {a['by_kind'].get('persona_edit', 0)}회, 블록 보상 발행 {a['by_kind'].get('epoch_issued', 0)}회", ""]
    first, last = a["first"], a["last"]
    if last:
        out += ["## 경제와 평판", "",
                f"유통 {_p(last['supply']['circulating_mP'])} · 소각 누계 {_p(last['supply']['burned_mP'])} · "
                f"에스크로 {_p(last['supply']['escrow_mP'])} · 지니 계수 {last['gini']}"
                + (f" (기간 시작 {first['gini']})" if first else ""), "",
                "| 노드 | 잔액 | 변화 | 평판 | 변화 | 수입 누계 | 지식 노트 |", "|---|---|---|---|---|---|---|"]
        for n in sorted(last["balances_mP"]):
            b0 = first["balances_mP"].get(n) if first else None
            r0 = first["reputation"].get(n) if first else None
            b1, r1 = last["balances_mP"][n], last["reputation"][n]
            out.append(f"| {n} | {_p(b1)} | {_p(b1 - b0) if b0 is not None else '-'} | {r1:.3f} | "
                       f"{(r1 - r0):+.3f} | {_p(last['income_mP'].get(n, 0))} | {last['knowledge_notes'].get(n, 0)} |"
                       if r0 is not None else
                       f"| {n} | {_p(b1)} | - | {r1:.3f} | - | {_p(last['income_mP'].get(n, 0))} | {last['knowledge_notes'].get(n, 0)} |")
        out += ["", f"현상금 결과: {json.dumps(last['bounties'], ensure_ascii=False)} · "
                    f"분쟁: {json.dumps(last['disputes'], ensure_ascii=False)} · "
                    f"도구: {json.dumps(last['tools'], ensure_ascii=False)} · "
                    f"투표 일치율: {json.dumps(last['vote_agreement'], ensure_ascii=False) or '-'}", ""]
    out += ["## 노드별 활동", "", "| 노드 | 에피소드 (종류/종료) | 많이 쓴 도구 |", "|---|---|---|"]
    for n in sorted(set(a["episodes"]) | set(a["tools_used"])):
        eps = ", ".join(f"{k} {v}" for k, v in a["episodes"][n].most_common())
        tl = ", ".join(f"{k} {v}" for k, v in a["tools_used"][n].most_common(6))
        out.append(f"| {n} | {eps} | {tl} |")
    out += ["", "## 지식 체계", ""]
    for n in sorted(a["notes"]):
        out.append(f"### {n} (기간 중 새 웹 사실 {a['web_facts'].get(n, 0)}개)")
        for note in a["notes"][n]:
            d = note["data"]
            out.append(f"- **{note['topic']}** — 확신 {d.get('confidence')}, 사실 {len(d.get('facts', []))}개: "
                       f"{d.get('summary', '')[:300]}")
            if d.get("open_questions"):
                out.append(f"  - 더 알아볼 것: {' / '.join(d['open_questions'])}")
        out.append("")
    if a["domains"]:
        out += ["읽은 사이트 상위: " + ", ".join(f"{k} {v}" for k, v in a["domains"].most_common(10)), ""]
    out += ["## 성향 변화", ""]
    any_persona = False
    for n in sorted(a["persona"]):
        prev_text = None
        for p in a["persona"][n]:
            any_persona = True
            d = p["data"]
            verdict = {"approved": "승인", "rejected": "거부", "rollback": "롤백"}.get(d.get("verdict"), d.get("verdict"))
            out.append(f"### {n} v{d.get('version')} — {verdict} ({_fmt_ts(p['ts'])})")
            if d.get("reason"):
                out.append(f"이유: {d['reason']}")
            if d.get("failed"):
                out.append(f"실패한 검사: {d['failed']}")
            if d.get("verdict") in ("approved", "rollback"):
                base = prev_text
                if base is None:
                    from aichain.mastermind import prompt
                    persona_name = {"n1": "seeker", "n2": "artisan", "n3": "warden", "n4": "nomad"}.get(n)
                    base = prompt.load(persona_name) if persona_name else ""
                diff = [l for l in difflib.unified_diff(base.splitlines(), d.get("text", "").splitlines(),
                                                        lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
                out += ["```diff", *diff[:30], "```"]
                prev_text = d.get("text")
            out.append("")
    if not any_persona:
        out += ["(성향 수정 없음)", ""]
    out += ["## 이상과 오류", ""]
    out += [f"- {k}: {v}" for k, v in a["problems"].most_common()] or ["- 없음"]
    out += ["", "---", f"생성: {time.strftime('%Y-%m-%d %H:%M')} · `python -m aichain.report`"]
    return "\n".join(out)


def render_html(md: str, title: str) -> str:
    """Small Markdown subset -> HTML (headings, lists, tables, code, bold)."""
    lines, out, in_table, in_code, in_list = md.splitlines(), [], False, False, False

    def inline(t: str) -> str:
        t = html.escape(t)
        import re
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", t)

    for line in lines:
        if line.startswith("```"):
            out.append("</pre>" if in_code else '<pre class="diff">')
            in_code = not in_code
            continue
        if in_code:
            cls = "add" if line.startswith("+") else "del" if line.startswith("-") else ""
            out.append(f'<span class="{cls}">{html.escape(line)}</span>')
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-"} for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.lstrip().startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line.lstrip()[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fff; --ink:#1d2330; --muted:#6b7385; --line:#e3e6ec; --add:#1f8a4c; --del:#c0392b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171d; --ink:#e6e9ef; --muted:#9099ab; --line:#2a313c; --add:#4cc584; --del:#ff6b5e; }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;
  max-width:920px; margin:0 auto; padding:24px 16px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0; font-size:14px; }}
th,td {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; }}
pre {{ background:rgba(127,127,127,.1); padding:8px; border-radius:6px; overflow:auto; font-size:13px; }}
.add {{ color:var(--add); display:block; }} .del {{ color:var(--del); display:block; }}
h1 {{ font-size:22px; }} h2 {{ font-size:18px; margin-top:28px; border-bottom:1px solid var(--line); }} h3 {{ font-size:15px; }}
code {{ font-size:13px; }} p, li {{ word-break: break-word; }}
</style></head><body>{"".join(out)}</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AIChain experiment report")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--journal", type=Path, default=ROOT / "data" / "journal" / "journal.db")
    ap.add_argument("--knowledge", type=Path, default=ROOT / "data" / "knowledge" / "knowledge.db")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reports")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    since = None if args.all else time.time() - args.hours * 3600
    analysis = analyse(collect(args.journal, args.knowledge, since))
    label = "전체 기간" if args.all else (f"최근 {args.hours:g}시간" if args.hours != 24 else "일일")
    title = args.title or f"AIChain 실험 리포트 — {label} ({time.strftime('%Y-%m-%d')})"
    md = render_markdown(analysis, title)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"report-{time.strftime('%Y%m%d-%H%M')}" + ("-all" if args.all else f"-{int(args.hours)}h")
    (args.out / f"{stem}.md").write_text(md, encoding="utf-8")
    (args.out / f"{stem}.html").write_text(render_html(md, title), encoding="utf-8")
    print(args.out / f"{stem}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
