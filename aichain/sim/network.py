"""In-process network: one board, one content store, N nodes, a virtual block clock.

Stands in for P2P + root PoA (milestones M0–M1) so the Mastermind loop can be tested
end to end. One block = 5 seconds (720 blocks per hour, 02 doc).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

from aichain.board import Board, BoardError, BoardEvent, Params
from aichain.reputation import EPOCH_BLOCKS, Reputation, for_board
from aichain.content import ContentStore
from aichain.gateways.llm.base import LLMGateway
from aichain.gateways.web import SearchGateway
from aichain.memory import KnowledgeStore
from aichain.workshop.registry import Workshop
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
        self.board = Board(params)
        self.content = ContentStore()
        self.chain_stats: dict[str, int] | None = None
        self.content_author: dict[str, str] = {}
        self.nodes: dict[str, Node] = {}
        if chain is not None:
            for h, author, obj in chain.all_content():
                self.content.put(obj)
                if author:
                    self.content_author.setdefault(h, author)
            self.chain_stats = attach_chain(self.board, chain)
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
            self.board.open_account(cfg.node_id, grant_mP)
        self.nodes[cfg.node_id] = node
        if self.chain is not None:
            node.restore_from_chain()
        return node

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
