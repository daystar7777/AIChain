"""Read-only web viewer for the journal.

    python -m aichain.journal.viewer --db data/journal/journal.db   # http://127.0.0.1:8790

Opens the database read-only and never writes. Binds to loopback unless --allow-public.
"""

import argparse
import asyncio
import ipaddress
import json
import socket
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from aichain.journal.store import connect

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_DB = Path("data/journal/journal.db")
DEFAULT_KNOWLEDGE = Path("data/knowledge/knowledge.db")


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    if "data" in d:
        d["data"] = json.loads(d["data"])
    if "meta" in d:
        d["meta"] = json.loads(d["meta"])
    return d


def create_app(db_path: Path, poll_seconds: float = 0.5, keepalive_seconds: float = 15.0,
               knowledge_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="AIChain Journal", version="0.0.1", docs_url=None, redoc_url=None)

    def db() -> sqlite3.Connection:
        if not db_path.exists():
            raise HTTPException(503, f"journal database not found: {db_path}")
        return connect(db_path, readonly=True)

    def query(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        conn = db()
        try:
            return [_row(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def entries_sql(node: str | None, kind: str | None) -> tuple[str, list[Any]]:
        sql = "SELECT id, ts, height, node, kind, episode, data FROM entries WHERE run_id = ? AND id > ?"
        args: list[Any] = []
        if node:
            sql += " AND node IN (%s)" % ",".join("?" * len(node.split(",")))
            args += node.split(",")
        if kind:
            sql += " AND kind IN (%s)" % ",".join("?" * len(kind.split(",")))
            args += kind.split(",")
        return sql, args

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return query(
            "SELECT r.id, r.started_at, r.ended_at, r.label, r.meta, "
            "(SELECT COUNT(*) FROM entries e WHERE e.run_id = r.id) AS entries "
            "FROM runs r ORDER BY r.id DESC LIMIT 200")

    @app.get("/api/runs/{run_id}/entries")
    def entries(run_id: int, after: int = 0, node: str | None = None, kind: str | None = None,
                limit: Annotated[int, Query(ge=1, le=5000)] = 1000) -> list[dict[str, Any]]:
        sql, args = entries_sql(node, kind)
        return query(sql + " ORDER BY id LIMIT ?", (run_id, after, *args, limit))

    @app.get("/api/runs/{run_id}/state")
    def state(run_id: int) -> dict[str, Any]:
        rows = query("SELECT ts, data FROM snapshots WHERE run_id = ?", (run_id,))
        if not rows:
            raise HTTPException(404, "no snapshot for this run")
        return {"ts": rows[0]["ts"], **rows[0]["data"]}

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: int, limit: Annotated[int, Query(ge=1, le=2000)] = 500) -> list[dict[str, Any]]:
        rows = query("SELECT ts, height, data FROM metrics WHERE run_id = ? ORDER BY height", (run_id,))
        if len(rows) > limit:  # downsample evenly, keep the latest
            step = len(rows) / limit
            rows = [rows[int(i * step)] for i in range(limit - 1)] + [rows[-1]]
        return rows

    @app.get("/api/runs/{run_id}/episodes/{episode}")
    def episode(run_id: int, episode: str) -> list[dict[str, Any]]:
        rows = query("SELECT id, ts, height, node, kind, episode, data FROM entries "
                     "WHERE run_id = ? AND episode = ? ORDER BY id", (run_id, episode))
        if not rows:
            raise HTTPException(404, "episode not found")
        return rows

    @app.get("/api/knowledge")
    def knowledge(node: str | None = None, facts: Annotated[int, Query(ge=0, le=500)] = 50) -> dict[str, Any]:
        """Each node's current notes (latest version per topic) and recent web facts."""
        if knowledge_path is None or not knowledge_path.exists():
            return {"nodes": {}}
        conn = sqlite3.connect(f"file:{knowledge_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            where, args = ("WHERE node = ?", (node,)) if node else ("", ())
            rows = [dict(r) for r in conn.execute(
                f"SELECT node, seq, hash, ts, height, kind, source, trust, content, tags, supersedes, "
                f"topic, data FROM memories {where} ORDER BY node, seq", args)]
        finally:
            conn.close()
        out: dict[str, Any] = {}
        dead = {r["supersedes"] for r in rows if r["supersedes"]}
        for r in rows:
            n = out.setdefault(r["node"], {"notes": [], "facts": [], "memories": 0})
            n["memories"] += 1
            if r["kind"] == "note":
                versions = sum(1 for x in rows if x["node"] == r["node"] and x["kind"] == "note"
                               and x["topic"] == r["topic"] and x["seq"] <= r["seq"])
                if r["hash"] not in dead:
                    n["notes"].append({"topic": r["topic"], "seq": r["seq"], "ts": r["ts"],
                                       "height": r["height"], "version": versions,
                                       **(json.loads(r["data"]) if r["data"] else {})})
            elif r["kind"] == "fact" and r["source"].startswith("web:"):
                n["facts"].append({"seq": r["seq"], "ts": r["ts"], "content": r["content"],
                                   "url": r["source"][4:], "tags": json.loads(r["tags"])})
        for n in out.values():
            n["facts"] = n["facts"][-facts:][::-1] if facts else []
        return {"nodes": out}

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: int, request: Request, node: str | None = None, kind: str | None = None,
                     after: int = 0,
                     last_event_id: Annotated[str | None, Header()] = None) -> StreamingResponse:
        cursor = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
        sql, args = entries_sql(node, kind)

        async def events() -> AsyncIterator[str]:
            nonlocal cursor
            idle = 0.0
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    rows = query(sql + " ORDER BY id LIMIT 500", (run_id, cursor, *args))
                except HTTPException:
                    rows = []
                for r in rows:
                    cursor = r["id"]
                    yield f"id: {r['id']}\ndata: {json.dumps(r, ensure_ascii=False)}\n\n"
                if rows:
                    idle = 0.0
                    continue
                await asyncio.sleep(poll_seconds)
                idle += poll_seconds
                if idle >= keepalive_seconds:
                    idle = 0.0
                    yield ": keepalive\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def _bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(2048)
    sock.set_inheritable(True)
    return sock


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="AIChain journal viewer (read-only)")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    ap.add_argument("--host", action="append", default=None,
                    help="address to bind; repeat for several (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--allow-public", action="store_true",
                    help="allow non-loopback addresses (e.g. the Tailscale IP)")
    args = ap.parse_args(argv)
    hosts = args.host or ["127.0.0.1"]
    for host in hosts:
        try:
            loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if host in ("0.0.0.0", "::"):
            print("refusing to bind every interface; name the addresses explicitly", file=sys.stderr)
            return 2
        if not loopback and not args.allow_public:
            print(f"refusing to bind {host}: use 127.0.0.1 or pass --allow-public", file=sys.stderr)
            return 2
    sockets = [_bind(h, args.port) for h in hosts]
    config = uvicorn.Config(create_app(args.db, knowledge_path=args.knowledge), log_level="info")
    uvicorn.Server(config).run(sockets=sockets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
