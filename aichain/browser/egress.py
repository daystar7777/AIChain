"""Egress service: the guard proxy on a Unix socket, the browser's only way out.

    python -m aichain.browser.egress --uds /run/aichain-egress/egress.sock --block 46.250.255.147

Runs as its own user with the network; systemd also denies private ranges to it at
the kernel level (IPAddressDeny), as a second layer under the guard.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time

from aichain.browser.guard import EgressGuard
from aichain.browser.proxy import GuardProxy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AIChain egress proxy")
    ap.add_argument("--uds", required=True)
    ap.add_argument("--mode", default="660", help="socket permissions (octal)")
    ap.add_argument("--block", action="append", default=[], help="extra blocked address/network")
    args = ap.parse_args(argv)
    if os.geteuid() == 0:
        print("refusing to run as root", file=sys.stderr)
        return 2

    def on_block(target: str, reason: str) -> None:
        print(json.dumps({"ts": time.time(), "blocked": target, "reason": reason}), flush=True)

    async def run() -> None:
        proxy = GuardProxy(EgressGuard(extra_blocked=args.block), on_block=on_block)
        await proxy.start_unix(args.uds, int(args.mode, 8))
        print(f"egress proxy on {args.uds}", flush=True)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        await proxy.close()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
