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
from aichain.sim.factory import ROOT, SHORT_PARAMS, ConfigError, build_network, load_default_env

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
            net = build_network(args.provider, args.model, args.nomad_model, max_calls, params,
                                trace=trace, echo=not args.quiet)
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 2
        net.bridge_question("n1", "demo", **DEFAULT_QUESTION)
        net.run_until_idle()
        net.advance(params.commit_blocks + params.reveal_blocks + params.judge_blocks
                    + params.dispute_blocks + 2)

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
    print(f"LLM calls: {net.gateway.calls}, episodes: {len(net.episodes)}, trace: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
