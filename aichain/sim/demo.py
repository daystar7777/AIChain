"""Run four nodes in one process through one bounty cycle.

    uv run python -m aichain.sim.demo                      # scripted fake model
    uv run python -m aichain.sim.demo --provider opencode-go --max-calls 10

A bridge question goes to n1 (Seeker), which decides whether to post a bounty; the
others answer per their persona; the board runs commit → reveal → judge → settle.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from aichain.board import Params
from aichain.common import fmt_P
from aichain.gateways.llm.base import GatewayConfig, LLMGateway, ProviderAdapter
from aichain.gateways.llm.opencode_go import DEFAULT_MODEL, OpenCodeGoAdapter
from aichain.gateways.llm.scripted import ScriptedAdapter
from aichain.node import NodeConfig
from aichain.sim.brains import persona_brain
from aichain.sim.network import Network

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def default_nodes(provider: str, model: str, nomad_model: str | None) -> list[NodeConfig]:
    return [
        NodeConfig("n1", "seeker", provider, model, tags_interest=[]),
        NodeConfig("n2", "artisan", provider, model, tags_interest=["code", "python", "debugging"]),
        # Warden reviews settled bounties rather than answering; no bounty_new wake-ups (saves calls).
        NodeConfig("n3", "warden", provider, model, tags_interest=[]),
        NodeConfig("n4", "nomad", provider, nomad_model or model, temperature=0.7,
                   tags_interest=["*"]),
    ]


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

    # Working directory first (e.g. /opt/aichain on the server), then the repo root.
    load_env(Path.cwd() / ".env")
    load_env(ROOT / ".env")
    adapters: dict[str, ProviderAdapter] = {}
    if args.provider == "scripted":
        adapters["scripted"] = ScriptedAdapter(persona_brain)
        model = args.model or "scripted"
        max_calls = args.max_calls
    else:
        key = os.environ.get("OPENCODE_GO_API_KEY", "")
        if not key:
            print("OPENCODE_GO_API_KEY is not set. Put it in .env (see .env.example).", file=sys.stderr)
            return 2
        adapters["opencode-go"] = OpenCodeGoAdapter(key)
        model = args.model or DEFAULT_MODEL
        max_calls = args.max_calls if args.max_calls is not None else 10

    trace_path = args.trace or ROOT / "data" / f"trace-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    gateway = LLMGateway(adapters, GatewayConfig(max_calls=max_calls))
    # Short periods so one cycle takes a few hundred virtual blocks.
    params = Params(commit_blocks=24, reveal_blocks=12, judge_blocks=48, dispute_blocks=120)
    with trace_path.open("w", encoding="utf-8") as trace:
        net = Network(gateway, params, trace=trace, echo=not args.quiet)
        for cfg in default_nodes(args.provider, model, args.nomad_model):
            net.add_node(cfg)
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
    print(f"LLM calls: {gateway.calls}, episodes: {len(net.episodes)}, trace: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
