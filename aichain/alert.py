"""Telegram alert for a failed systemd unit (OnFailure=aichain-alert@%n.service).

    python -m aichain.alert --unit aichain-bridge.service [--env-file /opt/aichain/.env]

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_MANAGER_CHAT_ID from the env file (parsed here, never put
into the environment or onto a command line), asks systemd how the unit failed, and sends one
short message with the last log lines (passed through the journal's secret masking). The token
never appears in this program's output. At most one alert per unit every 30 minutes; the next one
says how many failures went unreported in between. Without a token it only prints a line.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path("/var/lib/aichain-alert/state.json")
QUIET_SECONDS = 30 * 60
FACTS = ("Result", "ExecMainCode", "ExecMainStatus", "NRestarts", "Restart", "ActiveState")
RESULTS = {"exit-code": "비정상 종료", "signal": "시그널", "core-dump": "크래시", "watchdog": "watchdog(응답 없음)",
           "oom-kill": "메모리 한도 초과", "timeout": "시간 초과", "start-limit-hit": "재시작 한도"}


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of an env file (the same format aichain.sim.factory.load_env reads)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def unit_facts(unit: str, run: Callable[[list[str]], str] = _run) -> dict[str, str]:
    text = run(["systemctl", "show", unit, *(f"--property={p}" for p in FACTS)])
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def recent_log(unit: str, lines: int = 8, run: Callable[[list[str]], str] = _run) -> str:
    from aichain.journal.store import redact

    text = run(["journalctl", "--unit", unit, "--lines", str(lines), "--output", "cat", "--no-pager"])
    return redact(text.strip())[-1200:]


def build_message(unit: str, facts: dict[str, str], log: str, missed: int = 0) -> str:
    code = facts.get("ExecMainCode", "")
    status = facts.get("ExecMainStatus", "")
    how = {"1": f"종료 코드 {status}", "2": f"시그널 {status}로 죽음", "3": f"시그널 {status}로 덤프"}.get(
        code, f"상태 {status}" if status else "")
    restarts = facts.get("NRestarts", "0")
    lines = [f"🚨 {unit} 실패 ({socket.gethostname()})",
             f"원인: {RESULTS.get(facts.get('Result', ''), facts.get('Result', '?'))}" + (f" · {how}" if how else "")
             + (f" · 자동 재시작 {restarts}회째" if restarts not in ("", "0") else "")]
    if missed:
        lines.append(f"(지난 알림 이후 {missed}번 더 실패)")
    if log:
        lines += ["최근 로그:", log]
    lines.append("systemd가 다시 시작합니다." if facts.get("Restart", "no") != "no" else "자동 재시작 없음.")
    lines.append(f"확인: journalctl -u {unit} -n 100")
    return "\n".join(lines)


def take_slot(state_path: Path, unit: str, now: float, quiet: float = QUIET_SECONDS) -> int | None:
    """Rate limit per unit. Returns the number of failures not reported since the last alert,
    or None when this one should stay quiet (it is counted for the next alert)."""
    try:
        state: dict[str, Any] = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    entry = state.setdefault(unit, {"last": 0.0, "missed": 0})
    if now - entry["last"] < quiet:
        entry["missed"] += 1
        missed = None
    else:
        missed, entry["last"], entry["missed"] = entry["missed"], now, 0
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state))
    except OSError:
        pass  # without a state file every failure is reported; better than none
    return missed


def send(token: str, chat_id: str, text: str,
         urlopen: Callable[..., Any] = urllib.request.urlopen) -> None:
    body = json.dumps({"chat_id": chat_id, "text": text[:3900]}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body,
                                 headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=20) as r:
        if not json.loads(r.read() or b"{}").get("ok"):
            raise RuntimeError("telegram refused the message")


def main(argv: list[str] | None = None, run: Callable[[list[str]], str] = _run,
         urlopen: Callable[..., Any] = urllib.request.urlopen) -> int:
    ap = argparse.ArgumentParser(description="Telegram alert for a failed AIChain systemd unit")
    ap.add_argument("--unit", required=True)
    ap.add_argument("--env-file", type=Path, default=ROOT / ".env")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = ap.parse_args(argv)
    env = {**read_env(args.env_file), **{k: v for k, v in os.environ.items() if k.startswith("TELEGRAM_")}}
    token, chat = env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_MANAGER_CHAT_ID", "")
    if not token or not chat:
        print(f"{args.unit} failed; no alert sent (TELEGRAM_BOT_TOKEN / TELEGRAM_MANAGER_CHAT_ID not set "
              f"in {args.env_file})")
        return 0
    missed = take_slot(args.state, args.unit, time.time())
    if missed is None:
        print(f"{args.unit} failed again; alert suppressed (one per {QUIET_SECONDS // 60} min)")
        return 0
    text = build_message(args.unit, unit_facts(args.unit, run), recent_log(args.unit, run=run), missed)
    try:
        send(token, chat, text, urlopen)
    except (OSError, urllib.error.URLError, ValueError, RuntimeError) as e:
        print(f"telegram alert for {args.unit} failed: {type(e).__name__}: {str(e).replace(token, '***')}",
              file=sys.stderr)
        return 1
    print(f"alert sent for {args.unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
