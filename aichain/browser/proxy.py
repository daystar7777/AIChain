"""A tiny forward proxy that is the browser's only way out.

Handles CONNECT (HTTPS) and absolute-URI HTTP requests. Each request is checked by
`EgressGuard`, then connected to the very address that passed the check.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from urllib.parse import urlsplit

from aichain.browser.guard import Blocked, EgressGuard

MAX_HEADER_BYTES = 64 * 1024


class GuardProxy:
    def __init__(self, guard: EgressGuard, on_block: Callable[[str, str], None] | None = None,
                 connect_timeout: float = 10.0) -> None:
        self.guard = guard
        self.on_block = on_block or (lambda target, reason: None)
        self.connect_timeout = connect_timeout
        self._server: asyncio.base_events.Server | None = None
        self.port: int | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self._server = await asyncio.start_server(self._handle, host, port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def start_unix(self, path: str, mode: int = 0o660) -> None:
        """Listen on a Unix socket (the egress service; the browser has no network)."""
        import os

        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        self._server = await asyncio.start_unix_server(self._handle, path)
        os.chmod(path, mode)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
            writer.close()
            return
        if len(head) > MAX_HEADER_BYTES:
            await self._reply(writer, 431, "headers too large")
            return
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, version = lines[0].split(" ", 2)
        except ValueError:
            await self._reply(writer, 400, "bad request line")
            return
        try:
            if method.upper() == "CONNECT":
                host, _, port_s = target.rpartition(":")
                port = int(port_s)
                addresses = await self.guard.resolve_checked(host, port)
                upstream = await self._connect(addresses, port)
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                await self._pipe(reader, writer, *upstream)
            else:
                parts = urlsplit(target)
                host, port = self.guard.split_url(target)
                addresses = await self.guard.resolve_checked(host, port)
                if parts.scheme != "http":
                    raise Blocked("only http may be proxied without CONNECT")
                upstream = await self._connect(addresses, port)
                path = parts.path or "/"
                if parts.query:
                    path += "?" + parts.query
                headers = [h for h in lines[1:] if h and not h.lower().startswith(("proxy-", "connection:"))]
                request = f"{method} {path} {version}\r\n" + "\r\n".join(headers) + "\r\nConnection: close\r\n\r\n"
                upstream[1].write(request.encode("latin-1"))
                await upstream[1].drain()
                await self._pipe(reader, writer, *upstream)
        except Blocked as e:
            self.on_block(target, str(e))
            await self._reply(writer, 403, f"blocked: {e}")
        except (OSError, asyncio.TimeoutError, ValueError) as e:
            await self._reply(writer, 502, f"upstream error: {e}")

    async def _connect(self, addresses: list[str], port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        last: Exception | None = None
        for addr in addresses:
            try:
                return await asyncio.wait_for(asyncio.open_connection(addr, port),
                                              timeout=self.connect_timeout)
            except (OSError, asyncio.TimeoutError) as e:
                last = e
        raise OSError(f"could not connect: {last}")

    async def _pipe(self, client_r, client_w, up_r, up_w) -> None:
        async def copy(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while data := await src.read(65536):
                    dst.write(data)
                    await dst.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    dst.close()

        await asyncio.gather(copy(client_r, up_w), copy(up_r, client_w))

    async def _reply(self, writer: asyncio.StreamWriter, status: int, message: str) -> None:
        body = message.encode()
        reason = {400: "Bad Request", 403: "Forbidden", 431: "Request Header Fields Too Large",
                  502: "Bad Gateway"}.get(status, "Error")
        with contextlib.suppress(Exception):
            writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\n"
                         f"X-AIChain-Blocked: {status == 403}\r\nContent-Length: {len(body)}\r\n"
                         f"Connection: close\r\n\r\n".encode() + body)
            await writer.drain()
            writer.close()


class UnixForwarder:
    """Inside the browser's private network namespace: 127.0.0.1:<port> → egress Unix socket.

    Chromium can only use a TCP proxy, and its namespace has nothing but loopback, so
    this forwarder is the browser's one path to the outside.
    """

    def __init__(self, uds_path: str) -> None:
        self.uds_path = uds_path
        self._server: asyncio.base_events.Server | None = None
        self.port: int | None = None

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_r, up_w = await asyncio.open_unix_connection(self.uds_path)
        except OSError:
            writer.close()
            return

        async def copy(src, dst):
            try:
                while data := await src.read(65536):
                    dst.write(data)
                    await dst.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    dst.close()

        await asyncio.gather(copy(reader, up_w), copy(up_r, writer))
