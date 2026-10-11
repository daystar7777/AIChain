"""Run the network with the Bridge API in front of it.

    uv run python -m aichain.bridge.tokens new my-client      # prints a token once
    uv run python -m aichain.bridge                          # scripted model, 127.0.0.1:8780
    uv run python -m aichain.bridge --provider opencode-go --max-calls 40

Binds to loopback only unless --allow-public is given (expose it behind nginx later).
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
import time
from pathlib import Path

import uvicorn

from aichain.board import Params
from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import DEFAULT_FILE, TokenStore
from aichain.sim.factory import ROOT, SHORT_PARAMS, ConfigError, build_network, load_default_env


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["scripted", "opencode-go"], default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--nomad-model", default=None)
    ap.add_argument("--max-calls", type=int, default=None, help="hard cap on total LLM calls")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--allow-public", action="store_true", help="allow binding a non-loopback address")
    ap.add_argument("--block-seconds", type=float, default=5.0, help="real seconds per block (design: 5)")
    ap.add_argument("--periods", choices=["short", "design"], default="short",
                    help="short: ~17 min per bounty cycle at 5 s/block; design: 03 doc defaults")
    ap.add_argument("--tokens", type=Path, default=DEFAULT_FILE)
    args = ap.parse_args(argv)

    if not _is_loopback(args.host) and not args.allow_public:
        print(f"refusing to bind {args.host}: use 127.0.0.1 or pass --allow-public", file=sys.stderr)
        return 2
    load_default_env()
    tokens = TokenStore.load(args.tokens)
    if not tokens.clients:
        print(f"no clients in {args.tokens}; create one with: python -m aichain.bridge.tokens new <client>",
              file=sys.stderr)
        return 2
    trace_path = ROOT / "data" / f"bridge-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    params = SHORT_PARAMS if args.periods == "short" else Params()
    with trace_path.open("w", encoding="utf-8", buffering=1) as trace:
        try:
            net = build_network(args.provider, args.model, args.nomad_model, args.max_calls, params,
                                trace=trace)
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 2
        runtime = NetworkRuntime(net, block_seconds=args.block_seconds)
        print(f"trace: {trace_path}", file=sys.stderr)
        uvicorn.run(create_app(runtime, tokens), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
