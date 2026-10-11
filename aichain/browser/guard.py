"""Egress guard for the headless browser (SSRF defence, design doc 05 1.2).

A URL is allowed only if it is http/https on port 80/443 and *every* address its host
resolves to is a public unicast address outside the blocked ranges. The browser
reaches the web only through `GuardProxy`, which checks each request (redirects and
sub-resources included) and then connects to the exact address it checked, so a DNS
answer that changes between check and use (rebinding) cannot sneak through.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

BLOCKED_NETWORKS = [ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10",  # 100.64/10: CGNAT, includes Tailscale
    "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4",
    "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8", "64:ff9b::/96", "2001:db8::/32",
)]

Resolver = Callable[[str, int], Awaitable[list[str]]]


class Blocked(Exception):
    """The request is not allowed. The message says why."""


async def system_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


class EgressGuard:
    def __init__(
        self,
        extra_blocked: Iterable[str] = (),
        allowed_ports: Iterable[int] = (80, 443),
        resolver: Resolver = system_resolver,
        allow_hosts: Iterable[str] = (),
    ) -> None:
        self.blocked = BLOCKED_NETWORKS + [ipaddress.ip_network(n, strict=False) for n in extra_blocked]
        self.allowed_ports = set(allowed_ports)
        self.resolver = resolver
        # Test hook only: hosts that skip the address check (never set in production).
        self.allow_hosts = set(allow_hosts)

    def check_ip(self, ip: str | IPAddress) -> IPAddress:
        addr = ipaddress.ip_address(ip) if isinstance(ip, str) else ip
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        if any(addr in net for net in self.blocked if net.version == addr.version):
            raise Blocked(f"address {addr} is in a blocked range")
        if not addr.is_global:
            raise Blocked(f"address {addr} is not a public address")
        return addr

    def split_url(self, url: str) -> tuple[str, int]:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise Blocked(f"scheme {parts.scheme or '(none)'} is not allowed")
        if not parts.hostname:
            raise Blocked("URL has no host")
        if parts.username or parts.password:
            raise Blocked("credentials in URLs are not allowed")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return parts.hostname, port

    def check_port(self, port: int) -> None:
        if port not in self.allowed_ports:
            raise Blocked(f"port {port} is not allowed")

    async def resolve_checked(self, host: str, port: int) -> list[str]:
        """Resolve and validate; returns the addresses that may be connected to."""
        self.check_port(port)
        host = host.strip("[]").rstrip(".").lower()
        if host in self.allow_hosts:
            return await self.resolver(host, port)
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise Blocked(f"host {host} is internal")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            return [str(self.check_ip(literal))]
        try:
            addresses = await self.resolver(host, port)
        except OSError as e:
            raise Blocked(f"cannot resolve {host}: {e}") from e
        if not addresses:
            raise Blocked(f"{host} has no addresses")
        for a in addresses:  # every answer must be public, not just the first
            self.check_ip(a)
        return addresses

    def precheck(self, url: str) -> None:
        """Checks that need no DNS (used where the browser has no network of its own)."""
        host, port = self.split_url(url)
        self.check_port(port)
        host = host.strip("[]").rstrip(".").lower()
        if host in self.allow_hosts:
            return
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise Blocked(f"host {host} is internal")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            return
        self.check_ip(literal)

    async def check_url(self, url: str) -> list[str]:
        host, port = self.split_url(url)
        return await self.resolve_checked(host, port)
