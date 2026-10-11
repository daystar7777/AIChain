"""Read-only web viewer for the journal.

    python -m aichain.journal.viewer --db data/journal/journal.db   # http://127.0.0.1:8790

Opens the database read-only and never writes. Binds to loopback unless --allow-public.
"""

import argparse
import asyncio
import ipaddress
import json
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from aichain.journal.store import connect

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_DB = Path("data/journal/journal.db")


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    if "data" in d:
        d["data"] = json.loads(d["data"])
    if "meta" in d:
        d["meta"] = json.loads(d["meta"])
    return d


def create_app(db_path: Path, poll_seconds: float = 0.5, keepalive_seconds: float = 15.0) -> FastAPI:
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

    @app.get("/api/runs/{run_id}/episodes/{episode}")
    def episode(run_id: int, episode: str) -> list[dict[str, Any]]:
        rows = query("SELECT id, ts, height, node, kind, episode, data FROM entries "
                     "WHERE run_id = ? AND episode = ? ORDER BY id", (run_id, episode))
        if not rows:
            raise HTTPException(404, "episode not found")
        return rows

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


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="AIChain journal viewer (read-only)")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--allow-public", action="store_true")
    args = ap.parse_args(argv)
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not args.allow_public:
        print(f"refusing to bind {args.host}: use 127.0.0.1 or pass --allow-public", file=sys.stderr)
        return 2
    uvicorn.run(create_app(args.db), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
