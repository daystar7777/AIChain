"""In-process network: one board, one content store, N nodes, a virtual block clock.

Stands in for P2P + root PoA (milestones M0–M1) so the Mastermind loop can be tested
end to end. One block = 5 seconds (720 blocks per hour, 02 doc).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

from aichain.board import Board, BoardEvent, Params
from aichain.content import ContentStore
from aichain.gateways.llm.base import LLMGateway
from aichain.gateways.web import SearchGateway
from aichain.memory import KnowledgeStore
from aichain.workshop.registry import Workshop
from aichain.node import EpisodeResult, Node, NodeConfig

SECONDS_PER_BLOCK = 5


class Network:
    def __init__(self, gateway: LLMGateway, params: Params | None = None,
                 trace: TextIO | None = None, echo: bool = False,
                 web: SearchGateway | None = None, knowledge: KnowledgeStore | None = None,
                 research_interval_blocks: int | None = None,
                 workshop: Workshop | None = None) -> None:
        self.gateway = gateway
        self.workshop = workshop
        self.web = web
        self.knowledge = knowledge
        # When set, idle nodes with interests get a research_tick every this many blocks.
        self.research_interval_blocks = research_interval_blocks
        self.board = Board(params)
        self.content = ContentStore()
        self.content_author: dict[str, str] = {}
        self.nodes: dict[str, Node] = {}
        self.events: list[dict[str, Any]] = []
        self.episodes: list[EpisodeResult] = []
        self._trace = trace
        self._echo = echo
        # Called with every log entry (the Bridge follows question lifecycles this way).
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        # Set by Node.run_episode so every entry logged during an episode carries its id.
        self.current_episode: str | None = None

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
            listener(entry)

    def add_node(self, cfg: NodeConfig, grant_mP: int = 1_000_000) -> Node:
        node = Node(cfg, self)
        self.board.open_account(cfg.node_id, grant_mP)
        self.nodes[cfg.node_id] = node
        return node

    def put_content(self, obj: Any, author: str) -> str:
        h = self.content.put(obj)
        self.content_author.setdefault(h, author)
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

    def run_until_idle(self, max_rounds: int = 50) -> None:
        for _ in range(max_rounds):
            pending = [n for n in self.nodes.values() if n.has_pending]
            if not pending:
                return
            for node in pending:
                self.episodes.extend(node.run_pending())
        raise RuntimeError("network did not go idle")

    def research(self, node_ids: list[str] | None = None) -> None:
        for node in self.nodes.values():
            if (node_ids is None or node.id in node_ids) and node.cfg.interests:
                node.enqueue("research_tick", {})

    def advance(self, blocks: int) -> None:
        for _ in range(blocks):
            events = self.board.advance(1)
            if events:
                self.dispatch(events)
            n = self.research_interval_blocks
            if n and self.board.height % n == 0:
                self.research([nd.id for nd in self.nodes.values() if not nd.has_pending])
            for node in self.nodes.values():
                node.run_duties()
            self.run_until_idle()

    def balances(self) -> dict[str, dict[str, int]]:
        return {nid: {"balance_mP": a.balance_mP, "locked_mP": a.locked_mP}
                for nid, a in self.board.accounts.items()}
