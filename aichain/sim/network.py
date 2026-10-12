"""In-process network: one board, one content store, N nodes, a virtual block clock.

Stands in for P2P + root PoA (milestones M0–M1) so the Mastermind loop can be tested
end to end. One block = 5 seconds (720 blocks per hour, 02 doc).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

from aichain.board import Board, BoardError, BoardEvent, Params
from aichain.common import sha256_hex
from aichain.reputation import EPOCH_BLOCKS, Reputation, for_board
from aichain.content import ContentStore
from aichain.gateways.llm.base import LLMGateway
from aichain.gateways.web import SearchGateway
from aichain.memory import KnowledgeStore
from aichain.workshop.registry import Workshop
from aichain.chain.crypto import KeyStore
from aichain.chain.store import ChainStore, attach as attach_chain
from aichain.node import EpisodeResult, Node, NodeConfig

SECONDS_PER_BLOCK = 5


class Network:
    def __init__(self, gateway: LLMGateway, params: Params | None = None,
                 trace: TextIO | None = None, echo: bool = False,
                 web: SearchGateway | None = None, knowledge: KnowledgeStore | None = None,
                 research_interval_blocks: int | None = None,
                 workshop: Workshop | None = None,
                 chain: ChainStore | None = None) -> None:
        self.gateway = gateway
        self.chain = chain
        self.workshop = workshop
        self.web = web
        self.knowledge = knowledge
        # When set, idle nodes with interests get a research_tick every this many blocks.
        self.research_interval_blocks = research_interval_blocks
        # Daily reflection (04 doc 5.2), staggered per node; None disables.
        self.reflect_interval_blocks: int | None = None
        self.board = Board(params)
        self.content = ContentStore()
        self.chain_stats: dict[str, int] | None = None
        # Account keys: signing, and end-to-end encryption of private deliveries.
        self.keys: KeyStore = KeyStore(None)
        self.content_author: dict[str, str] = {}
        self.nodes: dict[str, Node] = {}
        if chain is not None:
            for h, author, obj in chain.all_content():
                self.content.put(obj)
                if author:
                    self.content_author.setdefault(h, author)
            self.keys = KeyStore(chain.path.parent / "keys")
            self.chain_stats = attach_chain(self.board, chain, self.keys)
        self.events: list[dict[str, Any]] = []
        self.episodes: list[EpisodeResult] = []
        self._trace = trace
        self._echo = echo
        # Called with every log entry (the Bridge follows question lifecycles this way).
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        # Set by Node.run_episode so every entry logged during an episode carries its id.
        self.current_episode: str | None = None
        self.listener_errors = 0
        self._rep_cache: dict[str, tuple] = {}

    def clock_minutes(self) -> float:
        return self.board.height * SECONDS_PER_BLOCK / 60

    def log(self, node: str, event: str, **data: Any) -> None:
        entry = {"height": self.board.height, "node": node, "event": event, **data}
        if self.current_episode and "episode" not in entry:
            entry["episode"] = self.current_episode
        self.events.append(entry)
        line = json.dumps(entry, ensure_ascii=False)
        if self._trace:
            self._trace.write(line + "\n")
        if self._echo:
            print(line, file=sys.stdout, flush=True)
        for listener in self.listeners:
            try:
                listener(entry)
            except Exception as e:  # e.g. journal disk full: never let observers stop the network
                self.listener_errors += 1
                if self.listener_errors <= 5 or self.listener_errors % 100 == 0:
                    print(f"[aichain] listener {getattr(listener, '__qualname__', listener)} failed "
                          f"({self.listener_errors}x): {type(e).__name__}: {e}", file=sys.stderr)

    def add_node(self, cfg: NodeConfig, grant_mP: int = 1_000_000) -> Node:
        node = Node(cfg, self)
        if cfg.node_id not in self.board.accounts:  # already there when the chain was replayed
            pub = self.keys.pubkey(cfg.node_id)
            self.board.open_account(cfg.node_id, grant_mP, pub)
        self.nodes[cfg.node_id] = node
        if self.chain is not None:
            node.restore_from_chain()
        return node

    def announce_proposal(self, proposal_id: str) -> None:
        prop = self.board.proposals[proposal_id]
        for node in self.nodes.values():
            if node.id != prop.proposer and node.id in self.board.voters() \
                    and self.gateway.has_daily_room(node.id):
                node.enqueue("proposal_open", {"proposal_id": proposal_id})

    def complete_purchase(self, purchase_id: str) -> str | None:
        """Core-to-Core delivery of an info purchase, end-to-end encrypted:
        seller Core seals the plaintext to the buyer's key -> buyer Core opens it, checks it against
        the offer's commitment and confirms (or the deal goes to the operator). No LLM involved."""
        from aichain.chain.crypto import DecryptError, seal_for

        pu = self.board.purchases[purchase_id]
        offer = self.board.offers[pu.offer_id]
        seller, buyer = self.nodes.get(pu.seller), self.nodes.get(pu.buyer)
        if seller is None or buyer is None:
            return None
        plaintext = seller.private.get(f"offer:{offer.offer_id}")
        if plaintext is None:
            return None  # cannot deliver; the buyer is refunded at the deadline
        ciphertext = seal_for(self.board.account(pu.buyer).pubkey, plaintext.encode())
        c_hash = self.put_content({"sealed_for": pu.buyer, "box": ciphertext}, author=pu.seller)
        self.board.deliver_offer(pu.seller, purchase_id, c_hash)
        try:
            opened = self.keys.open_sealed(pu.buyer, ciphertext).decode()
            ok = sha256_hex(opened) == offer.content_commit
        except (DecryptError, UnicodeDecodeError):
            opened, ok = None, False
        reason = self.put_content({"check": "commitment", "ok": ok}, author=pu.buyer)
        self.board.confirm_delivery(pu.buyer, purchase_id, ok, reason)
        self.log("core", "purchase_delivered", purchase_id=purchase_id, offer_id=offer.offer_id,
                 seller=pu.seller, buyer=pu.buyer, price=pu.price_mP / 1000, verified=ok)
        if not ok:
            return None
        buyer.memory.remember(self.board.height, "fact", f"deal:{pu.seller}",
                              round(self.reputation(pu.seller).value, 3),
                              f"[구매 {offer.title}] {opened[:1500]}", ["deal", offer.title[:40]],
                              [c_hash], data={"offer_id": offer.offer_id, "purchase_id": purchase_id})
        return opened

    def announce_tool_request(self, request_id: str) -> None:
        req = self.board.tool_requests[request_id]
        for node in self.nodes.values():
            if node.id != req.buyer and "tool_submit" in node.tool_names() \
                    and self.gateway.has_daily_room(node.id):
                node.enqueue("tool_request_new", {"request_id": request_id})

    def verify_tool_submission(self, request_id: str, tool_id: str, version: int) -> Any:
        """Core runs the buyer's acceptance tests on the submitted version in the isolated
        executor; root records the result on chain (objective, no LLM judgement)."""
        req = self.board.tool_requests[request_id]
        tests = self.content.get(req.tests_hash) or []
        try:
            passed, report = self.workshop.run_tests(tool_id, version, tests)
        except Exception as e:  # unknown tool, execution disabled, executor failure
            passed, report = False, [{"error": f"{type(e).__name__}: {e}"[:300]}]
        report_hash = self.put_content(report, author="core")
        req = self.board.verify_submission("root", request_id, tool_id, version, passed, report_hash)
        self.log("core", "tool_submission_verified", request_id=request_id, tool_id=tool_id,
                 version=version, passed=passed, state=req.state, winner=req.winner)
        if req.state == "FULFILLED" and req.winner_tool == [tool_id, version]:
            for nid in (req.buyer, req.winner):
                if nid in self.nodes:
                    self.nodes[nid].memory.remember(
                        self.board.height, "episode", f"chain:{request_id}", 1.0,
                        f"도구 구매 요청 {request_id} '{req.title}' 체결: {req.winner}의 {tool_id} v{version}, "
                        f"보상 {req.reward_mP / 1000:g}P", tags=["tool", "trade"])
        return req

    def notify_dispute(self, dispute_id: str) -> None:
        """Core memory for both parties (no LLM call): opened or resolved."""
        d = self.board.disputes[dispute_id]
        text = (f"분쟁 {d.dispute_id} ({d.target_kind} {d.target_id}): 제기 {d.opener}, 피고 {d.defendant}, "
                f"상태 {d.state}")
        for nid in (d.opener, d.defendant):
            if nid in self.nodes:
                self.nodes[nid].memory.remember(self.board.height, "episode", f"chain:{d.dispute_id}",
                                                1.0, text, tags=["dispute"])
        self.log("core", "dispute_state", dispute_id=d.dispute_id, state=d.state,
                 opener=d.opener, defendant=d.defendant)

    def reputation(self, account: str) -> Reputation:
        """Deterministic reputation (03 doc 4.3), cached until height or events change."""
        key = (account, self.board.height, len(self.board.rep_events))
        cached = self._rep_cache.get(account)
        if cached is None or cached[0] != key:
            cached = (key, for_board(self.board, account))
            self._rep_cache[account] = cached
        return cached[1]

    def put_content(self, obj: Any, author: str) -> str:
        h = self.content.put(obj)
        self.content_author.setdefault(h, author)
        if self.chain is not None:
            self.chain.put_content(h, obj, author)
        return h

    # ---- routing ----

    def bridge_question(self, node_id: str, client: str, title: str, body: str,
                        acceptance_criteria: str, **extra: Any) -> None:
        """extra: ref, max_reward_P, deadline_minutes (set by the Bridge API)."""
        self.nodes[node_id].enqueue("bridge_question", {
            "client": client, "title": title, "body": body,
            "acceptance_criteria": acceptance_criteria, **extra})

    def link_question(self, ref: str, bounty_id: str) -> None:
        self.log("core", "bridge_linked", ref=ref, bounty_id=bounty_id)

    def announce_bounty(self, bounty_id: str) -> None:
        b = self.board.bounty(bounty_id)
        for node in self.nodes.values():
            if node.id == b.asker:
                continue
            interest = set(node.cfg.tags_interest)
            if "*" in interest or interest & set(b.tags):
                node.enqueue("bounty_new", {"bounty_id": bounty_id})

    def dispatch(self, events: list[BoardEvent]) -> None:
        for ev in events:
            self.log("board", ev.kind, **ev.payload)
            if ev.kind == "judge_due" and ev.to in self.nodes:
                self.nodes[ev.to].enqueue("judge_due", ev.payload)
            elif ev.kind == "bounty_settled":
                for nid in ev.payload["participants"]:
                    if nid in self.nodes:
                        self.nodes[nid].on_settled(ev.payload)
                if ev.payload["outcome"] in ("PAID", "AUTO_SPLIT", "REFUNDED"):
                    for node in self.nodes.values():
                        if node.cfg.watch_settlements and node.id not in ev.payload["participants"] \
                                and self.gateway.has_daily_room(node.id):
                            node.enqueue("review_settled", {"bounty_id": ev.payload["bounty_id"]})
            elif ev.kind in ("proposal_tallied", "proposal_approved"):
                prop = self.board.proposals[ev.payload["proposal_id"]]
                for nid in self.board.voters():
                    if nid in self.nodes:
                        self.nodes[nid].memory.remember(
                            self.board.height, "episode", f"chain:{prop.proposal_id}", 1.0,
                            f"제안 {prop.proposal_id} '{prop.title}' → {prop.state} {prop.tally or ''}",
                            tags=["governance"])
            elif ev.kind == "tool_request_expired":
                req = self.board.tool_requests[ev.payload["request_id"]]
                if req.buyer in self.nodes:
                    self.nodes[req.buyer].memory.remember(
                        self.board.height, "episode", f"chain:{req.request_id}", 1.0,
                        f"도구 구매 요청 {req.request_id} '{req.title}' 마감: 통과작 없음, 보상 환불", tags=["tool"])
            elif ev.kind == "epoch_issued":
                pass  # logged above; balances already credited
            elif ev.kind == "dispute_resolved":
                self.notify_dispute(ev.payload["dispute_id"])

    # ---- running ----

    @property
    def halted(self) -> str | None:
        return self.board.halted

    def run_until_idle(self, max_rounds: int = 50) -> bool:
        """Run queued episodes. Returns False if events were still pending after max_rounds
        (an event storm); they stay queued and run on the next call instead of crashing."""
        for _ in range(max_rounds):
            if self.halted:
                return False
            pending = [n for n in self.nodes.values() if n.has_pending]
            if not pending:
                return True
            for node in pending:
                self.episodes.extend(node.run_pending())
        self.log("core", "busy", reason=f"events still pending after {max_rounds} rounds")
        return False

    def reflect(self, node_ids: list[str] | None = None) -> None:
        for node in self.nodes.values():
            if node_ids is not None and node.id not in node_ids:
                continue
            if not self.gateway.has_daily_room(node.id):
                continue
            marker = next((r for r in reversed(node.memory.records)
                           if r.kind == "reflection" and r.source == "chain:reflect"), None)
            node.enqueue("reflect", {"since_seq": marker.seq if marker else 0,
                                     "prev": (marker.data or {}) if marker else {},
                                     "period": "지난 회고 이후" if marker else "시작 이후"})
            node.memory.remember(self.board.height, "reflection", "chain:reflect", 1.0,
                                 f"회고 시점: 평판 {self.reputation(node.id).value:.2f}, "
                                 f"잔액 {self.board.account(node.id).balance_mP / 1000:g}P",
                                 data={"rep": self.reputation(node.id).value,
                                       "balance": self.board.account(node.id).balance_mP})

    def research(self, node_ids: list[str] | None = None) -> None:
        for node in self.nodes.values():
            if (node_ids is None or node.id in node_ids) and node.cfg.interests:
                if not self.gateway.has_daily_room(node.id):
                    continue  # no LLM calls left today: don't queue work that cannot run
                node.enqueue("research_tick", {})

    def advance(self, blocks: int) -> None:
        for _ in range(blocks):
            if self.halted:
                return
            events = self.board.advance(1)
            if events:
                self.dispatch(events)
            r = self.reflect_interval_blocks
            if r and self.nodes:
                ids = sorted(self.nodes)
                offset = r // len(ids)
                due = [nid for i, nid in enumerate(ids) if (self.board.height + i * offset) % r == 0]
                if due:
                    self.reflect(due)
            n = self.research_interval_blocks
            if n and self.board.height % n == 0:
                self.research([nd.id for nd in self.nodes.values() if not nd.has_pending])
            if self.board.height % EPOCH_BLOCKS == 0 and not self.halted:
                for node in self.nodes.values():  # Core duty, no LLM (04 doc 6.1)
                    try:
                        self.board.heartbeat(node.id, self.board.height // EPOCH_BLOCKS)
                    except BoardError:
                        pass
            for node in self.nodes.values():
                node.run_duties()
            self.run_until_idle()

    def balances(self) -> dict[str, dict[str, int]]:
        return {nid: {"balance_mP": a.balance_mP, "locked_mP": a.locked_mP}
                for nid, a in self.board.accounts.items()}
