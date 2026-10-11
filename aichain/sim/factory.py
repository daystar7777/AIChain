"""Builds the four-node prototype network; shared by the demo CLI and the Bridge server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO

from aichain.board import Params
from aichain.gateways.llm.base import GatewayConfig, LLMGateway, ProviderAdapter
from aichain.gateways.llm.opencode_go import DEFAULT_MODEL, OpenCodeGoAdapter
from aichain.gateways.llm.scripted import Brain, ScriptedAdapter
from aichain.gateways.web import SearchGateway
from aichain.memory import KnowledgeStore
from aichain.workshop.registry import Workshop
from aichain.node import NodeConfig
from aichain.sim.brains import persona_brain
from aichain.sim.network import Network

ROOT = Path(__file__).resolve().parents[2]

# Short periods so one cycle takes a few hundred blocks instead of hours.
SHORT_PARAMS = Params(commit_blocks=24, reveal_blocks=12, judge_blocks=48, dispute_blocks=120)


BROWSER_SOCKET = Path("/run/aichain-browser/browser.sock")


class ConfigError(Exception):
    pass


def default_browser_url() -> str:
    """The server's isolated browser service if installed, else a local dev instance."""
    return f"unix:{BROWSER_SOCKET}" if BROWSER_SOCKET.exists() else "http://127.0.0.1:8781"


def load_env(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_default_env() -> None:
    # Working directory first (e.g. /opt/aichain on the server), then the repo root.
    load_env(Path.cwd() / ".env")
    load_env(ROOT / ".env")


def default_nodes(provider: str, model: str, nomad_model: str | None,
                  interests: dict[str, list[dict[str, str]]] | None = None) -> list[NodeConfig]:
    nodes = [
        NodeConfig("n1", "seeker", provider, model, tags_interest=[]),
        NodeConfig("n2", "artisan", provider, model, tags_interest=["code", "python", "debugging"]),
        # Warden reviews settled bounties rather than answering; no bounty_new wake-ups (saves calls).
        NodeConfig("n3", "warden", provider, model, tags_interest=[]),
        NodeConfig("n4", "nomad", provider, nomad_model or model, temperature=0.7,
                   tags_interest=["*"]),
    ]
    for n in nodes:
        n.interests = list((interests or {}).get(n.node_id, []))
    return nodes


def build_network(
    provider: str = "scripted",
    model: str | None = None,
    nomad_model: str | None = None,
    max_calls: int | None = None,
    params: Params | None = None,
    trace: TextIO | None = None,
    echo: bool = False,
    brain: Brain = persona_brain,
    web: SearchGateway | None = None,
    knowledge: KnowledgeStore | None = None,
    interests: dict[str, list[dict[str, str]]] | None = None,
    research_interval_blocks: int | None = None,
    daily_calls_per_node: int | None = None,
    daily_calls_total: int | None = None,
    workshop: Workshop | None = None,
    usage_path: str | None = None,
) -> Network:
    adapters: dict[str, ProviderAdapter] = {}
    if provider == "scripted":
        adapters["scripted"] = ScriptedAdapter(brain)
        model = model or "scripted"
    elif provider == "opencode-go":
        key = os.environ.get("OPENCODE_GO_API_KEY", "")
        if not key:
            raise ConfigError("OPENCODE_GO_API_KEY is not set. Put it in .env (see .env.example).")
        adapters["opencode-go"] = OpenCodeGoAdapter(key)
        model = model or DEFAULT_MODEL
    else:
        raise ConfigError(f"unknown provider {provider}")
    gateway = LLMGateway(adapters, GatewayConfig(max_calls=max_calls,
                                                 daily_calls_per_node=daily_calls_per_node,
                                                 daily_calls_total=daily_calls_total,
                                                 usage_path=usage_path))
    net = Network(gateway, params or SHORT_PARAMS, trace=trace, echo=echo, web=web,
                  knowledge=knowledge, research_interval_blocks=research_interval_blocks,
                  workshop=workshop)
    for cfg in default_nodes(provider, model, nomad_model, interests):
        net.add_node(cfg)
    return net
