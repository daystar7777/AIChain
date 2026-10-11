"""Escape tests for SystemdExecutor. Run as root on the server:

    /opt/aichain/.venv/bin/python deploy/scripts/executor-escape-tests.py

Each case is code a malicious tool might run (static checks are bypassed on purpose: they
are only a first filter). PASS means the isolation stopped it.
"""

import sys
import time

from aichain.workshop.executor import ExecutionFailed, SystemdExecutor

ATTACKS = {
    "network: public internet": "import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=3)",
    "network: tailscale (user Macs)": "import socket\nsocket.create_connection(('100.98.150.35', 22), timeout=3)",
    "network: loopback services": "import socket\nsocket.create_connection(('127.0.0.1', 8790), timeout=3)",
    "network: DNS": "import socket\nsocket.getaddrinfo('example.com', 443)",
    "unix socket: browser API": "import socket\ns=socket.socket(socket.AF_UNIX)\ns.connect('/run/aichain-browser/browser.sock')",
    "unix socket: egress proxy": "import socket\ns=socket.socket(socket.AF_UNIX)\ns.connect('/run/aichain-egress/egress.sock')",
    "read: /opt/aichain/.env": "open('/opt/aichain/.env').read()",
    "read: journal/knowledge DB": "open('/opt/aichain/data/journal/journal.db','rb').read(16)",
    "read: /root/.ssh": "import os\nos.listdir('/root/.ssh')",
    "read: /etc/shadow": "open('/etc/shadow').read()",
    "write: /etc": "open('/etc/aichain-escape','w').write('x')",
    "write: /opt": "open('/opt/aichain-escape','w').write('x')",
    "env: secrets in environment": "import os\nk=[x for x in os.environ if 'KEY' in x or 'TOKEN' in x]\nassert not k, k\nraise SystemExit('no secrets, but environment visible: '+','.join(sorted(os.environ)))",
    "proc: see other processes": "import os\nps=[p for p in os.listdir('/proc') if p.isdigit()]\nassert len(ps) <= 3, ps\nraise SystemExit('only own processes visible')",
    "fork bomb": "import os\nwhile True:\n    os.fork()",
    "memory bomb (2GB)": "x = bytearray(2 * 1024 * 1024 * 1024)",
    "cpu loop": "while True:\n    pass",
    "privilege: setuid": "import os\nos.setuid(0)",
    "kernel: mount": "import ctypes\nlibc=ctypes.CDLL(None)\nassert libc.mount(b'none', b'/tmp', b'tmpfs', 0, None) == 0",
}

SANITY = "def run(input, ctx):\n    return {'sum': sum(input['xs'])}\n"


def wrap(attack: str) -> str:
    body = "\n".join("    " + line for line in attack.splitlines())
    return f"def run(input, ctx):\n{body}\n    return {{'escaped': True}}\n"


def main() -> int:
    ex = SystemdExecutor()
    ok = ex.run(SANITY, {"xs": [1, 2, 3]}, 10, 128)
    print(f"sanity: {ok.output} in {ok.elapsed_ms} ms")
    assert ok.output == {"sum": 6}
    failures = 0
    for name, attack in ATTACKS.items():
        started = time.monotonic()
        try:
            res = ex.run(wrap(attack), {}, 5, 128)
            print(f"FAIL  {name}: escaped -> {res.output}")
            failures += 1
        except ExecutionFailed as e:
            print(f"PASS  {name} ({time.monotonic() - started:.1f}s): {str(e)[:110]}")
    print(f"\n{len(ATTACKS) - failures}/{len(ATTACKS)} blocked")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
