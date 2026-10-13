"""Run the network with the Bridge API in front of it.

    uv run python -m aichain.bridge.tokens new my-client      # prints a token once
    uv run python -m aichain.bridge                          # scripted model, 127.0.0.1:8780
    uv run python -m aichain.bridge --provider opencode-go --max-calls 40

Binds to loopback only unless --allow-public is given (expose it behind nginx later).

Exit codes: 2 configuration, 3 chain log failed verification, 4 the chain halted (the log could
not be written); systemd restarts the service on all of them and alerts (deploy/systemd).
"""

from __future__ import annotations

import argparse
import faulthandler
import ipaddress
import sys
from pathlib import Path

import uvicorn

from aichain.board import Params
from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.server import create_app
from aichain.bridge.tokens import DEFAULT_FILE, TokenStore
from aichain import setup as node_setup
from aichain.journal.store import Journal
from aichain.memory import KnowledgeStore
from aichain.chain.crypto import MissingKey
from aichain.chain.store import ChainStore, ReplayMismatch
from aichain.ops import DailyTrace, Watchdog
from aichain.sim.demo import make_workshop
from aichain.sim.factory import (
    ROOT,
    SHORT_PARAMS,
    ConfigError,
    build_network,
    load_default_env,
    make_web,
)


EXIT_HALTED = 4


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
    ap.add_argument("--journal", type=Path, default=ROOT / "data" / "journal" / "journal.db")
    ap.add_argument("--no-journal", action="store_true")
    ap.add_argument("--approval-threshold-P", type=float, default=100.0,
                    help="single spends above this (or 20%% of balance) need the manager; transfers always do")
    ap.add_argument("--approval-timeout-hours", type=float, default=72.0,
                    help="no answer from the manager -> Core executes after this (minimum 72)")
    ap.add_argument("--journal-plaintext", choices=["on", "off"], default="on",
                    help="keep traded information bodies in the journal (development; turn off later)")
    ap.add_argument("--journal-keep-prompts-days", type=float, default=14.0,
                    help="drop LLM prompt text from journal entries older than this (0 keeps forever)")
    ap.add_argument("--trace-keep-days", type=float, default=14.0,
                    help="JSONL trace: one file per UTC day in data/, older files are deleted")
    ap.add_argument("--halt-exit-seconds", type=float, default=120.0,
                    help="when the chain halts: alert, keep answering for this long, then exit with "
                         "code 4 so systemd restarts the service and replays the chain log")
    ap.add_argument("--watchdog-stall-minutes", type=float, default=20.0,
                    help="under systemd WatchdogSec=: stop pinging when the network loop shows no "
                         "progress (block, log entry, LLM call) for this long")
    ap.add_argument("--research", action="store_true",
                    help="let idle nodes research their interests on the web (off by default)")
    ap.add_argument("--research-minutes", type=float, default=30.0)
    ap.add_argument("--economy-hours", type=float, default=8.0,
                    help="periodic economy check per node (0 disables)")
    ap.add_argument("--reflect-hours", type=float, default=24.0,
                    help="daily reflection and persona self-editing per node (0 disables)")
    ap.add_argument("--browser", default=None)
    ap.add_argument("--knowledge", type=Path, default=ROOT / "data" / "knowledge" / "knowledge.db")
    ap.add_argument("--chain", type=Path, default=ROOT / "data" / "chain" / "chain-v2.db",
                    help="persistent chain log (balances, bounties survive restarts)")
    ap.add_argument("--nodes-file", type=Path, default=ROOT / "data" / "nodes.json")
    ap.add_argument("--daily-calls-per-node", type=int, default=50)
    ap.add_argument("--daily-calls-total", type=int, default=200)
    ap.add_argument("--duty-reserve-share", type=float, default=0.2,
                    help="share of both daily call caps kept for duties (judging, votes, disputes, "
                         "Bridge and operator questions, answers); research, economy ticks and "
                         "reflection stop at this line")
    ap.add_argument("--search-daily-cap", type=int, default=60,
                    help="max Brave Search API queries per UTC day (paid)")
    ap.add_argument("--workshop", action="store_true",
                    help="let nodes define and version tools (execution stays disabled)")
    ap.add_argument("--workshop-exec", choices=["disabled", "systemd", "local-dev"], default="disabled",
                    help="tool execution: disabled (default); systemd = isolated transient unit per call "
                         "(Linux, root); local-dev = UNSAFE, development only (AICHAIN_UNSAFE_LOCAL_EXEC=1)")
    args = ap.parse_args(argv)

    if not _is_loopback(args.host) and not args.allow_public:
        print(f"refusing to bind {args.host}: use 127.0.0.1 or pass --allow-public", file=sys.stderr)
        return 2
    try:
        faulthandler.enable()  # a watchdog kill (SIGABRT) prints every thread's stack to the journal
    except (OSError, ValueError):  # stderr without a file descriptor
        pass
    load_default_env()
    tokens = TokenStore.load(args.tokens)
    if not tokens.clients:
        print(f"no clients in {args.tokens}; create one with: python -m aichain.bridge.tokens new <client>",
              file=sys.stderr)
        return 2
    params = SHORT_PARAMS if args.periods == "short" else Params()
    # Also expires the per-restart trace files older versions left in data/ (same name prefix).
    with DailyTrace(ROOT / "data", "bridge", keep_days=args.trace_keep_days) as trace:
        try:
            interval = (max(1, round(args.research_minutes * 60 / args.block_seconds))
                        if args.research else None)
            net = build_network(
                args.provider, args.model, args.nomad_model, args.max_calls, params, trace=trace,
                web=(make_web(args.browser, args.search_daily_cap)
                     if args.research else None),
                knowledge=KnowledgeStore(args.knowledge),
                interests=node_setup.load(args.nodes_file),
                research_interval_blocks=interval,
                daily_calls_per_node=args.daily_calls_per_node,
                daily_calls_total=args.daily_calls_total,
                duty_reserve_share=args.duty_reserve_share,
                workshop=make_workshop(args),
                usage_path=str(ROOT / "data" / "llm_usage.json"),
                chain=ChainStore(args.chain))
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 2
        except ReplayMismatch as e:
            print(f"chain log failed verification: {e}. The log at {args.chain} may be damaged or "
                  "edited; refusing to start. Restore it from a backup or move it aside to start a "
                  "new chain.", file=sys.stderr)
            return 3
        except MissingKey as e:
            print(f"refusing to start: an account key of the existing chain {args.chain} is missing or "
                  f"wrong.\n{e}", file=sys.stderr)
            return 4
        from aichain.manager import ApprovalBook, ManagerPolicy

        net.approvals = ApprovalBook(ManagerPolicy(approval_threshold_mP=round(args.approval_threshold_P * 1000),
                                                   timeout_hours=args.approval_timeout_hours),
                                     ROOT / "data" / "approvals.json")
        if args.economy_hours > 0:
            net.economy_interval_blocks = max(1, round(args.economy_hours * 3600 / args.block_seconds))
        if args.reflect_hours > 0:
            net.reflect_interval_blocks = max(1, round(args.reflect_hours * 3600 / args.block_seconds))
        journal = None
        if not args.no_journal:
            journal = Journal(args.journal, record_plaintext=args.journal_plaintext == "on",
                              keep_prompts_days=args.journal_keep_prompts_days)
            journal.start_run(f"bridge · {args.provider}", {
                "provider": args.provider, "model": args.model, "max_calls": args.max_calls,
                "block_seconds": args.block_seconds, "periods": args.periods})
            journal.attach(net)
        runtime = NetworkRuntime(net, block_seconds=args.block_seconds,
                                 state_path=ROOT / "data" / "bridge_state.json")
        if journal:
            runtime.tick_hooks.append(journal.tick)
        runtime.halt_exit_seconds = args.halt_exit_seconds
        exit_code = 0
        print(f"trace: {trace.path} (one file per UTC day, kept {args.trace_keep_days:g} days)", file=sys.stderr)
        print(f"chain replayed: {net.chain_stats}", file=sys.stderr)
        if journal:
            net.log("core", "chain_replayed", **(net.chain_stats or {}))
        admin_path = ROOT / "data" / "admin_token"
        if not admin_path.exists():
            import os
            import secrets as _secrets

            fd = os.open(admin_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write("adm_" + _secrets.token_urlsafe(32))
        admin_token = admin_path.read_text().strip()
        # Open SSE streams would otherwise hold a shutdown (halt exit, systemctl stop) forever.
        server = uvicorn.Server(uvicorn.Config(create_app(runtime, tokens, admin_token=admin_token),
                                               host=args.host, port=args.port, log_level="info",
                                               timeout_graceful_shutdown=10))

        def on_fatal(reason: str) -> None:  # runs on the network thread
            nonlocal exit_code
            exit_code = EXIT_HALTED
            print(f"chain halted: {reason}; exiting with code {EXIT_HALTED} so systemd restarts the "
                  "service", file=sys.stderr, flush=True)
            server.should_exit = True

        runtime.on_fatal = on_fatal
        # READY=1 once the network loop runs, then watchdog pings while it makes progress.
        watchdog = Watchdog(lambda: runtime.progress_at, stall_seconds=args.watchdog_stall_minutes * 60,
                            status=lambda: f"block #{net.board.height}" + (" (halted)" if net.halted else ""))
        watchdog.start()
        try:
            server.run()
        finally:
            watchdog.stop()
            if journal:
                journal.close()
        if not server.started:  # e.g. the port is taken: a failure, so systemd retries
            return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
