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
import hmac
import json
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, Literal

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


class RulingIn(BaseModel):
    ruling: str = Field(pattern="^(uphold|reject|void)$")
    note: str = Field(default="", max_length=2000)


def create_app(runtime: NetworkRuntime, tokens: TokenStore, *, start_runtime: bool = True,
               sse_poll_seconds: float = 0.5, sse_keepalive_seconds: float = 15.0,
               admin_token: str | None = None) -> FastAPI:
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

    @app.exception_handler(TimeoutError)
    async def busy(request: Any, exc: TimeoutError) -> Any:
        # runtime.call waits for the simulation thread, which may be inside a long LLM episode.
        # The action stays queued and runs as soon as the thread is free.
        from fastapi.responses import JSONResponse

        return JSONResponse({"accepted": True, "detail": "the network is busy with an episode; the action "
                             "is queued and will run when it finishes"}, status_code=202)

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
        if runtime.net.halted:
            raise HTTPException(503, "network halted (chain log unavailable); operator must restart it")
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
                try:
                    events, done = runtime.events_since(ref, after)
                except KeyError:
                    return
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

    # ---- operator (root) endpoints: dispute rulings (03 doc 3.4) ----

    def operator(creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]) -> None:
        if not admin_token or creds is None or not hmac.compare_digest(creds.credentials, admin_token):
            raise HTTPException(401, "operator token required", headers={"WWW-Authenticate": "Bearer"})

    def dispute_view(net: Any, d: Any) -> dict[str, Any]:
        board = net.board
        target: dict[str, Any] = {}
        if d.target_kind in ("answer", "rejection") and d.target_id in board.answers:
            a = board.answers[d.target_id]
            b = board.bounty(a.bounty_id)
            target = {"bounty_id": b.bounty_id, "outcome": str(b.outcome),
                      "question": net.content.get(b.question_hash),
                      "answer": net.content.get(a.body_hash) if a.body_hash else None,
                      "author": a.author, "accepted": b.accepted == a.answer_id,
                      "rejected_reason": a.rejected_reason}
        elif d.target_kind == "rating" and d.target_id in board.ratings:
            r = board.ratings[d.target_id]
            target = {"rater": r.rater, "subject": r.subject, "score": r.score, "context_ref": r.context_ref}
        return {"dispute_id": d.dispute_id, "state": d.state, "target_kind": d.target_kind,
                "target_id": d.target_id, "opener": d.opener, "defendant": d.defendant,
                "bond_mP": d.bond_mP, "opened_height": d.opened_height, "deadline": d.deadline,
                "height": board.height, "claim": net.content.get(d.claim_hash), "target": target}

    @app.get("/admin/v1/disputes", dependencies=[Depends(operator)])
    def list_disputes(state: str | None = "open") -> list[dict[str, Any]]:
        return runtime.call(lambda net: [dispute_view(net, d) for d in net.board.disputes.values()
                                         if state in (None, "all") or d.state == state])

    @app.post("/admin/v1/disputes/{dispute_id}/resolve", dependencies=[Depends(operator)])
    def resolve(dispute_id: str, body: RulingIn) -> dict[str, Any]:
        def act(net: Any) -> dict[str, Any]:
            from aichain.board import BoardError

            note_hash = net.put_content({"ruling": body.ruling, "note": body.note}, author="operator")
            try:
                d = net.board.resolve_dispute("root", dispute_id, body.ruling, note_hash)
            except BoardError as e:
                raise HTTPException(409, str(e)) from e
            net.log("operator", "dispute_resolved", dispute_id=dispute_id, ruling=body.ruling,
                    note=body.note[:300])
            net.notify_dispute(dispute_id)
            return dispute_view(net, d)

        return runtime.call(act)

    class VoteIn(BaseModel):
        choice: str = Field(pattern="^(yes|no|abstain)$")
        reason: str = Field(min_length=3, max_length=1000)

    class VetoIn(BaseModel):
        reason: str = Field(min_length=3, max_length=1000)

    def proposal_view(net: Any, prop: Any) -> dict[str, Any]:
        from dataclasses import asdict

        return {**asdict(prop), "spec": net.content.get(prop.spec_hash), "height": net.board.height,
                "voters": net.board.voters()}

    @app.get("/admin/v1/proposals", dependencies=[Depends(operator)])
    def list_proposals(state: str | None = "open") -> list[dict[str, Any]]:
        open_states = ("VOTING", "TIMELOCK")
        return runtime.call(lambda net: [
            proposal_view(net, x) for x in net.board.proposals.values()
            if state in (None, "all") or (state == "open" and x.state in open_states) or x.state == state])

    def _gov_action(fn: Any) -> Any:
        def act(net: Any) -> Any:
            from aichain.board import BoardError
            try:
                return fn(net)
            except BoardError as e:
                raise HTTPException(409, str(e)) from e
        return runtime.call(act)

    @app.post("/admin/v1/proposals/{proposal_id}/vote", dependencies=[Depends(operator)])
    def operator_vote(proposal_id: str, body: VoteIn) -> dict[str, Any]:
        def fn(net: Any) -> dict[str, Any]:
            h = net.put_content({"reason": body.reason}, author="operator")
            net.board.vote("operator", proposal_id, body.choice, h)
            net.log("operator", "voted", proposal_id=proposal_id, choice=body.choice, reason=body.reason[:300])
            return proposal_view(net, net.board.proposals[proposal_id])
        return _gov_action(fn)

    @app.post("/admin/v1/proposals/{proposal_id}/veto", dependencies=[Depends(operator)])
    def root_veto(proposal_id: str, body: VetoIn) -> dict[str, Any]:
        def fn(net: Any) -> dict[str, Any]:
            h = net.put_content({"veto": body.reason}, author="root")
            net.board.veto_proposal("root", proposal_id, h)
            net.log("operator", "proposal_vetoed", proposal_id=proposal_id, reason=body.reason[:300])
            return proposal_view(net, net.board.proposals[proposal_id])
        return _gov_action(fn)

    class CloseIn(BaseModel):
        outcome: str = Field(pattern="^(implemented|abandoned)$")
        note: str = Field(min_length=3, max_length=1000)

    @app.post("/admin/v1/proposals/{proposal_id}/close", dependencies=[Depends(operator)])
    def root_close(proposal_id: str, body: CloseIn) -> dict[str, Any]:
        """Close an approved policy/upgrade proposal once it is implemented (or dropped)."""
        def fn(net: Any) -> dict[str, Any]:
            h = net.put_content({"outcome": body.outcome, "note": body.note}, author="root")
            net.board.close_proposal("root", proposal_id, body.outcome, h)
            net.log("operator", "proposal_closed", proposal_id=proposal_id, outcome=body.outcome,
                    note=body.note[:300])
            return proposal_view(net, net.board.proposals[proposal_id])
        return _gov_action(fn)

    class BlockToolIn(BaseModel):
        version: int
        reason: str = Field(min_length=3, max_length=1000)

    @app.post("/admin/v1/tools/{tool_id}/block", dependencies=[Depends(operator)])
    def root_block_tool(tool_id: str, body: BlockToolIn) -> dict[str, Any]:
        """Operator block (05 doc 4.3): withdrawn, no new licenses, paid calls refused."""
        def fn(net: Any) -> dict[str, Any]:
            h = net.put_content({"block": body.reason}, author="root")
            net.board.block_tool("root", tool_id, body.version, h)
            net.log("operator", "tool_blocked", tool_id=tool_id, version=body.version, reason=body.reason[:300])
            return {"tool_id": tool_id, "version": body.version, "blocked": True}
        return _gov_action(fn)

    class RollbackIn(BaseModel):
        version: int
        note: str = Field(default="", max_length=500)

    @app.get("/admin/v1/persona/{node_id}", dependencies=[Depends(operator)])
    def persona_history(node_id: str) -> dict[str, Any]:
        def fn(net: Any) -> dict[str, Any]:
            node = net.nodes.get(node_id)
            if node is None:
                raise HTTPException(404, "unknown node")
            return {"node": node_id, "current_version": node.persona_version,
                    "current_text": node.persona_text, "history": node.persona_history()}
        return runtime.call(fn)

    @app.post("/admin/v1/persona/{node_id}/rollback", dependencies=[Depends(operator)])
    def persona_rollback(node_id: str, body: RollbackIn) -> dict[str, Any]:
        def fn(net: Any) -> dict[str, Any]:
            node = net.nodes.get(node_id)
            if node is None:
                raise HTTPException(404, "unknown node")
            return {"node": node_id, "version": node.rollback_persona(body.version, body.note)}
        return _gov_action(fn)

    class DecisionIn(BaseModel):
        decision: str = Field(pattern="^(approve|deny)$")
        note: str = Field(default="", max_length=500)

    @app.get("/admin/v1/approvals", dependencies=[Depends(operator)])
    def list_approvals(state: str | None = "PENDING") -> list[dict[str, Any]]:
        from dataclasses import asdict

        def fn(net: Any) -> list[dict[str, Any]]:
            if net.approvals is None:
                return []
            return [asdict(a) for a in net.approvals.items.values() if state in (None, "all") or a.state == state]
        return runtime.call(fn)

    @app.post("/admin/v1/approvals/{approval_id}", dependencies=[Depends(operator)])
    def decide_approval(approval_id: str, body: DecisionIn) -> dict[str, Any]:
        from dataclasses import asdict

        def fn(net: Any) -> dict[str, Any]:
            if net.approvals is None:
                raise HTTPException(404, "approvals are not enabled")
            try:
                a = net.approvals.decide(approval_id, body.decision == "approve", "operator", body.note)
            except KeyError as e:
                raise HTTPException(409, str(e)) from e
            net.log("operator", "approval_decided", approval_id=approval_id, decision=body.decision, note=body.note)
            net.run_approvals()
            return asdict(a)
        return runtime.call(fn)

    class ManagerMsgIn(BaseModel):
        text: str = Field(min_length=1, max_length=4000)

    @app.post("/admin/v1/nodes/{node_id}/message", dependencies=[Depends(operator)])
    def message_node(node_id: str, body: ManagerMsgIn) -> dict[str, Any]:
        def fn(net: Any) -> dict[str, Any]:
            try:
                net.manager_receive(node_id, body.text)
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
            return {"queued": True}
        return runtime.call(fn)

    class TickIn(BaseModel):
        kind: Literal["economy_tick", "research_tick", "reflect"]

    @app.post("/admin/v1/nodes/{node_id}/tick", dependencies=[Depends(operator)])
    def tick_node(node_id: str, body: TickIn) -> dict[str, Any]:
        """Queue a scheduled event now (operator check that a node handles it), within its quota."""
        def fn(net: Any) -> dict[str, Any]:
            node = net.nodes.get(node_id)
            if node is None:
                raise HTTPException(404, f"no node {node_id}")
            if not net.gateway.has_daily_room(node_id):
                raise HTTPException(409, f"{node_id} has no LLM calls left today")
            node.enqueue(body.kind, {})
            return {"queued": body.kind}
        return runtime.call(fn)

    @app.get("/admin/v1/status", dependencies=[Depends(operator)])
    def admin_status() -> dict[str, Any]:
        return runtime.snapshot()

    @app.get("/admin/v1/outbox", dependencies=[Depends(operator)])
    def outbox(after: int = 0) -> list[dict[str, Any]]:
        return runtime.call(lambda net: [m for m in net.manager_outbox if m["id"] > after])

    @app.get("/bridge/v1/network")
    def network(client: Annotated[Client, Depends(client_of)]) -> dict[str, Any]:
        return runtime.snapshot()

    return app
