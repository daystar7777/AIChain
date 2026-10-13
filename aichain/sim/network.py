"""In-process network: one board, one content store, N nodes, a virtual block clock.

Stands in for P2P + root PoA (milestones M0–M1) so the Mastermind loop can be tested
end to end. One block = 5 seconds (720 blocks per hour, 02 doc).
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from typing import Any, Callable, TextIO

from aichain import audit
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
    # In-memory history is for tests and the demo; the journal and the trace keep everything.
    EVENTS_KEPT = 10_000
    EPISODES_KEPT = 1_000
    OUTBOX_KEPT = 500

    def __init__(self, gateway: LLMGateway, params: Params | None = None,
                 trace: TextIO | None = None, echo: bool = False,
                 web: SearchGateway | None = None, knowledge: KnowledgeStore | None = None,
                 research_interval_blocks: int | None = None,
                 workshop: Workshop | None = None,
                 chain: ChainStore | None = None) -> None:
        self.gateway = gateway
        self.chain = chain
        self.approvals = None  # ApprovalBook: outflows wait for the human manager
        self.audit_interval_blocks = audit.AUDIT_INTERVAL_BLOCKS
        # node -> manager messages (Telegram picks them up by id, so ids grow across restarts)
        self.manager_outbox: deque[dict[str, Any]] = deque(maxlen=self.OUTBOX_KEPT)
        self._outbox_seq = 0
        self.workshop = workshop
        self.web = web
        self.knowledge = knowledge
        # When set, idle nodes with interests get a research_tick every this many blocks.
        self.research_interval_blocks = research_interval_blocks
        # Daily reflection (04 doc 5.2), staggered per node; None disables.
        self.reflect_interval_blocks: int | None = None
        # Periodic economy check per node (staggered); None disables.
        self.economy_interval_blocks: int | None = None
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
        self.events: deque[dict[str, Any]] = deque(maxlen=self.EVENTS_KEPT)
        self.episodes: deque[EpisodeResult] = deque(maxlen=self.EPISODES_KEPT)
        self._trace = trace
        self._echo = echo
        self.trace_errors = 0
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
        try:
            if self._trace:
                self._trace.write(line + "\n")
            if self._echo:
                print(line, file=sys.stdout, flush=True)
        except (OSError, ValueError) as e:  # disk full, closed file or pipe: the trace is optional
            self.trace_errors += 1
            if self.trace_errors <= 5 or self.trace_errors % 1000 == 0:
                print(f"[aichain] trace write failed ({self.trace_errors}x): {type(e).__name__}: {e}",
                      file=sys.stderr)
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
        if offer.kind == "storage":  # lease: the quota follows the on-chain lease
            if seller.storage_free_kb() < offer.quantity:
                self.log("core", "rental_failed", purchase_id=purchase_id, reason="seller has no free space")
                return None
            self.board.deliver_offer(pu.seller, purchase_id, f"lease:storage:{offer.quantity}")
            reason = self.put_content({"check": "free space", "kb": offer.quantity}, author="core")
            self.board.confirm_delivery(pu.buyer, purchase_id, True, reason)
            self.log("core", "purchase_delivered", purchase_id=purchase_id, offer_id=offer.offer_id,
                     seller=pu.seller, buyer=pu.buyer, price=pu.price_mP / 1000, verified=True,
                     kind="storage", quantity=offer.quantity)
            return None
        if offer.kind != "info":  # quota rental: the gateway moves today's allowance
            try:
                self.gateway.transfer_quota(offer.kind, pu.seller, pu.buyer, offer.quantity)
            except Exception as e:  # seller used the quota meanwhile: refunded at the deadline
                self.log("core", "rental_failed", purchase_id=purchase_id, reason=str(e)[:200])
                return None
            self.board.deliver_offer(pu.seller, purchase_id, f"quota:{offer.kind}:{offer.quantity}")
            reason = self.put_content({"check": "quota transferred", "kind": offer.kind,
                                       "quantity": offer.quantity}, author="core")
            self.board.confirm_delivery(pu.buyer, purchase_id, True, reason)
            self.log("core", "purchase_delivered", purchase_id=purchase_id, offer_id=offer.offer_id,
                     seller=pu.seller, buyer=pu.buyer, price=pu.price_mP / 1000, verified=True,
                     kind=offer.kind, quantity=offer.quantity)
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

    def deliver_subscription(self, sub_id: str) -> str | None:
        """Core: send the seller's current note on the subscribed topic, sealed to the subscriber."""
        from aichain.chain.crypto import seal_for

        sub = self.board.subscriptions[sub_id]
        seller, buyer = self.nodes.get(sub.seller), self.nodes.get(sub.subscriber)
        if seller is None or buyer is None or sub.state != "ACTIVE":
            return None
        topic = seller.private.get(f"sub_topic:{sub.offer_id}")
        note = seller.memory.latest_note(topic) if topic else None
        if note is None or not note.data:
            return None
        text = json.dumps({"topic": note.topic, "version_seq": note.seq, **note.data}, ensure_ascii=False, indent=1)
        box = seal_for(self.board.account(sub.subscriber).pubkey, text.encode())
        c_hash = self.put_content({"sealed_for": sub.subscriber, "box": box}, author=sub.seller)
        self.board.sub_deliver(sub.seller, sub_id, c_hash)
        opened = self.keys.open_sealed(sub.subscriber, box).decode()
        buyer.memory.remember(self.board.height, "fact", f"deal:{sub.seller}",
                              round(self.reputation(sub.seller).value, 3),
                              f"[구독 {topic}] {opened[:1500]}", ["deal", "subscription", topic[:40]], [c_hash],
                              data={"sub_id": sub_id})
        self.log("core", "subscription_delivered", sub_id=sub_id, seller=sub.seller,
                 subscriber=sub.subscriber, topic=topic)
        return opened

    def announce_loan(self, loan_id: str) -> None:
        loan = self.board.loans[loan_id]
        for node in self.nodes.values():
            if node.id != loan.borrower and self.gateway.has_daily_room(node.id):
                node.enqueue("loan_request_new", {"loan_id": loan_id})

    def notify_loan(self, loan_id: str) -> None:
        loan = self.board.loans[loan_id]
        text = (f"대출 {loan_id}: 차입자 {loan.borrower}, 대여자 {loan.lender}, 원금 {loan.principal_mP / 1000:g}P, "
                f"이자 {loan.interest_mP / 1000:g}P, 상태 {loan.state}" + (f", 만기 #{loan.due}" if loan.due else ""))
        for nid in (loan.borrower, loan.lender):
            if nid in self.nodes:
                self.nodes[nid].memory.remember(self.board.height, "episode", f"chain:{loan_id}", 1.0, text,
                                                tags=["loan"])

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
            elif ev.kind == "loan_settled":
                self.notify_loan(ev.payload["loan_id"])
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

    def manager_send(self, node_id: str, text: str) -> dict[str, Any]:
        # Millisecond clock, so a restarted process never reuses an id the Telegram bridge has seen.
        self._outbox_seq = max(self._outbox_seq + 1, int(time.time() * 1000))
        item = {"id": self._outbox_seq, "node": node_id, "text": text[:2000],
                "height": self.board.height}
        self.manager_outbox.append(item)
        self.log(node_id, "manager_message", direction="to_manager", text=text[:500])
        return item

    def manager_receive(self, node_id: str, text: str, sender: str = "manager") -> None:
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"unknown node {node_id}")
        node.enqueue("manager_msg", {"text": text[:4000], "from": sender})
        self.log(node_id, "manager_message", direction="from_manager", text=text[:500])

    def run_approvals(self) -> None:
        """Execute approved outflows and those whose manager did not answer before the timeout."""
        if self.approvals is None:
            return
        for a in self.approvals.due():
            node = self.nodes.get(a.node)
            if node is None:
                continue
            result = node.execute_approval(a.args, a.tool)
            ok = not any(result.startswith(f'<data source="core">\n{p}')
                         for p in ("오류", "거부", "웹 오류", "내부 오류", "관리인 승인 대기"))
            self.approvals.finish(a, ok, result)
            self.log(a.node, "approval_executed", approval_id=a.approval_id, tool=a.tool, by=a.decided_by,
                     ok=ok, result=result[:300])
            node.memory.remember(self.board.height, "episode", f"manager:{a.approval_id}", 1.0,
                                 f"관리인 승인 건 {a.approval_id} ({a.tool}) {'집행' if ok else '실패'} "
                                 f"[{a.decided_by}]: {result[:300]}", ["approval"])
        for a in list(self.approvals.items.values()):
            if a.state == "DENIED" and not a.result:
                a.result = "denied"
                node = self.nodes.get(a.node)
                if node is not None:
                    node.memory.remember(self.board.height, "episode", f"manager:{a.approval_id}", 1.0,
                                         f"관리인이 {a.approval_id} ({a.tool})을 거부함: {a.note}", ["approval"])
                self.log(a.node, "approval_denied", approval_id=a.approval_id, note=a.note)

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
            e = self.economy_interval_blocks
            if e and self.nodes:
                ids = sorted(self.nodes)
                offset = e // len(ids)
                for i, nid in enumerate(ids):
                    node = self.nodes[nid]
                    if (self.board.height + i * offset + offset // 2) % e == 0 and not node.has_pending \
                            and self.gateway.has_daily_room(nid):
                        node.enqueue("economy_tick", {})
            n = self.research_interval_blocks
            if n and self.board.height % n == 0:
                self.research([nd.id for nd in self.nodes.values() if not nd.has_pending])
            if self.board.height % EPOCH_BLOCKS == 0 and not self.halted:
                for node in self.nodes.values():  # Core duty, no LLM (04 doc 6.1)
                    try:
                        self.board.heartbeat(node.id, self.board.height // EPOCH_BLOCKS)
                    except BoardError:
                        pass
            if self.audit_interval_blocks and self.board.height % self.audit_interval_blocks == 0:
                audit.run_on_network(self)
            for node in self.nodes.values():
                node.run_duties()
            self.run_approvals()
            self.run_until_idle()

    def balances(self) -> dict[str, dict[str, int]]:
        return {nid: {"balance_mP": a.balance_mP, "locked_mP": a.locked_mP}
                for nid, a in self.board.accounts.items()}
