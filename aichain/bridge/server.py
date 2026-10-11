"""Bridge API (design doc 04, ch. 7): the door through which external LLMs or people put
questions to the network.

    POST /bridge/v1/questions               submit a question → {ref, ...}
    GET  /bridge/v1/questions/{ref}         status, bounty, and the result once settled
    GET  /bridge/v1/questions/{ref}/events  Server-Sent Events stream of status changes
    GET  /bridge/v1/network                 public network statistics

Every endpoint needs `Authorization: Bearer <token>`; a client only sees its own
questions. The question text reaches the Mastermind as <data source="bridge:<client>">,
and the node decides by itself whether to post a bounty.
"""

# No `from __future__ import annotations`: FastAPI must evaluate the Annotated[...,
# Depends(local_function)] hints at definition time.
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from aichain.bridge.runtime import NetworkRuntime
from aichain.bridge.tokens import Client, TokenStore


class QuestionIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=8000)
    acceptance_criteria: str = Field(min_length=1, max_length=2000)
    max_reward_P: float | None = Field(default=None, ge=1, le=1000)
    deadline_minutes: int | None = Field(default=None, ge=1, le=1440)


_bearer = HTTPBearer(auto_error=False)


def create_app(runtime: NetworkRuntime, tokens: TokenStore, *, start_runtime: bool = True,
               sse_poll_seconds: float = 0.5, sse_keepalive_seconds: float = 15.0) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if start_runtime:
            runtime.start()
        try:
            yield
        finally:
            if start_runtime:
                runtime.stop()

    app = FastAPI(title="AIChain Bridge API", version="0.0.1", lifespan=lifespan)

    def client_of(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> Client:
        client = tokens.authenticate(creds.credentials) if creds else None
        if client is None:
            raise HTTPException(401, "invalid or missing bearer token",
                                headers={"WWW-Authenticate": "Bearer"})
        return client

    def own_question(ref: str, client: Client) -> dict[str, Any]:
        q = runtime.question(ref)
        # Same answer for "not yours" and "does not exist": refs are not enumerable.
        if q is None or q["client"] != client.name:
            raise HTTPException(404, "question not found")
        q.pop("client")
        return q

    @app.post("/bridge/v1/questions", status_code=202)
    def submit(q: QuestionIn, client: Annotated[Client, Depends(client_of)]) -> dict[str, Any]:
        if not tokens.take_quota(client):
            raise HTTPException(429, f"daily question limit ({client.daily_limit}) reached")
        rec = runtime.submit(client.name, q.title, q.body, q.acceptance_criteria,
                             q.max_reward_P, q.deadline_minutes)
        return {
            "ref": rec.ref,
            "status": rec.status,
            "node": rec.node,
            "links": {"self": f"/bridge/v1/questions/{rec.ref}",
                      "events": f"/bridge/v1/questions/{rec.ref}/events"},
        }

    @app.get("/bridge/v1/questions/{ref}")
    def get_question(ref: str, client: Annotated[Client, Depends(client_of)]) -> dict[str, Any]:
        return own_question(ref, client)

    @app.get("/bridge/v1/questions/{ref}/events")
    async def question_events(
        ref: str,
        request: Request,
        client: Annotated[Client, Depends(client_of)],
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        own_question(ref, client)
        try:
            after = int(last_event_id) if last_event_id else 0
        except ValueError:
            after = 0

        async def stream() -> AsyncIterator[str]:
            nonlocal after
            idle = 0.0
            yield "retry: 3000\n\n"
            while True:
                events, done = runtime.events_since(ref, after)
                for e in events:
                    after = e["id"]
                    payload = json.dumps(e, ensure_ascii=False)
                    yield f"id: {e['id']}\nevent: {e['event']}\ndata: {payload}\n\n"
                    idle = 0.0
                if done and not events:
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(sse_poll_seconds)
                idle += sse_poll_seconds
                if idle >= sse_keepalive_seconds:
                    idle = 0.0
                    yield ": keepalive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/bridge/v1/network")
    def network(client: Annotated[Client, Depends(client_of)]) -> dict[str, Any]:
        return runtime.snapshot()

    return app
