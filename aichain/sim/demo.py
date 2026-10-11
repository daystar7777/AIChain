"""Run four nodes in one process through one bounty cycle.

    uv run python -m aichain.sim.demo                      # scripted fake model
    uv run python -m aichain.sim.demo --provider opencode-go --max-calls 10

A bridge question goes to n1 (Seeker), which decides whether to post a bounty; the
others answer per their persona; the board runs commit → reveal → judge → settle.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from aichain.common import fmt_P
from aichain import setup as node_setup
from aichain.gateways.web import BrowserServiceProvider, SearchGateway
from aichain.journal.store import Journal
from aichain.memory import KnowledgeStore
from aichain.sim.factory import (
    ROOT,
    SHORT_PARAMS,
    ConfigError,
    build_network,
    default_browser_url,
    load_default_env,
)

DEFAULT_QUESTION = {
    "title": "Python 리스트를 제자리에서 뒤집는 방법",
    "body": "Python에서 새 리스트를 만들지 않고 기존 리스트를 제자리에서 뒤집는 가장 간단한 방법은? "
            "lst[::-1]과의 차이도 설명해 달라.",
    "acceptance_criteria": "제자리 뒤집기 방법을 제시하고, 새 리스트를 만드는 방법과의 차이를 "
                           "실행 가능한 예시로 보여줄 것.",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["scripted", "opencode-go"], default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--nomad-model", default=None, help="different model for n4 (design: N4 uses another model)")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="hard cap on total LLM calls (default: 10 for opencode-go, none for scripted)")
    ap.add_argument("--trace", type=Path, default=None, help="JSONL trace file (default: data/trace-<ts>.jsonl)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--journal", type=Path, default=ROOT / "data" / "journal" / "journal.db",
                    help="journal database for the web viewer")
    ap.add_argument("--no-journal", action="store_true")
    ap.add_argument("--scenario", choices=["bounty", "research", "both"], default="bounty",
                    help="bounty: one bridge question; research: one research episode per node")
    ap.add_argument("--research-nodes", default=None, help="comma-separated nodes that research (default all)")
    ap.add_argument("--browser", default=None, help="headless browser service")
    ap.add_argument("--no-web", action="store_true", help="run without web tools")
    ap.add_argument("--knowledge", type=Path, default=ROOT / "data" / "knowledge" / "knowledge.db",
                    help="persistent node memory (knowledge) database")
    ap.add_argument("--nodes-file", type=Path, default=ROOT / "data" / "nodes.json",
                    help="interests from `python -m aichain.setup`")
    ap.add_argument("--daily-calls-per-node", type=int, default=50)
    ap.add_argument("--daily-calls-total", type=int, default=200)
    args = ap.parse_args(argv)

    load_default_env()
    max_calls = args.max_calls
    if max_calls is None and args.provider == "opencode-go":
        max_calls = 10
    trace_path = args.trace or ROOT / "data" / f"trace-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    params = SHORT_PARAMS
    with trace_path.open("w", encoding="utf-8") as trace:
        try:
            web = None if args.no_web or args.scenario == "bounty" else SearchGateway(
                BrowserServiceProvider(args.browser or default_browser_url()))
            knowledge = KnowledgeStore(args.knowledge) if args.scenario != "bounty" else None
            net = build_network(args.provider, args.model, args.nomad_model, max_calls, params,
                                trace=trace, echo=not args.quiet, web=web, knowledge=knowledge,
                                interests=node_setup.load(args.nodes_file),
                                daily_calls_per_node=args.daily_calls_per_node,
                                daily_calls_total=args.daily_calls_total)
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 2
        journal = None
        if not args.no_journal:
            journal = Journal(args.journal)
            journal.start_run(f"demo · {args.provider}", {
                "provider": args.provider, "model": args.model, "max_calls": max_calls,
                "params": vars(params)})
            journal.attach(net)
        try:
            if args.scenario in ("research", "both"):
                nodes = args.research_nodes.split(",") if args.research_nodes else None
                net.research(nodes)
                net.run_until_idle()
            if args.scenario in ("bounty", "both"):
                net.bridge_question("n1", "demo", **DEFAULT_QUESTION)
                net.run_until_idle()
                net.advance(params.commit_blocks + params.reveal_blocks + params.judge_blocks
                            + params.dispute_blocks + 2)
        finally:
            if journal:
                journal.close()

    print("\n== result ==")
    for b in net.board.bounties.values():
        accepted = net.board.answers[b.accepted].author if b.accepted else "-"
        print(f"bounty {b.bounty_id}: {b.state} (outcome {b.outcome}), reward {fmt_P(b.reward_mP)}, "
              f"answers {len(b.answers)}, accepted {accepted}")
    if not net.board.bounties:
        print("no bounty was created")
    for nid, a in net.board.accounts.items():
        print(f"{nid}: balance {fmt_P(a.balance_mP)} locked {fmt_P(a.locked_mP)}")
    print(f"burned {fmt_P(net.board.burned_mP)}, supply check: {net.board.total_supply_check()}")
    for node in net.nodes.values():
        for note in node.memory.notes():
            if note.data:
                print(f"note {node.id} [{note.topic}] facts={len(note.data['facts'])} "
                      f"confidence={note.data['confidence']}: {note.data['summary'][:120]}")
    print(f"LLM calls: {net.gateway.calls}, episodes: {len(net.episodes)}, trace: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
