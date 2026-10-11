"""Prompt assembly: fixed core + persona layer (personas/*.md, from docs/personas.md)."""

from __future__ import annotations

from pathlib import Path

PERSONA_DIR = Path(__file__).resolve().parents[2] / "personas"

PERSONAS = ("seeker", "artisan", "warden", "nomad")


def load(name: str, persona_dir: Path = PERSONA_DIR) -> str:
    return (persona_dir / f"{name}.md").read_text(encoding="utf-8").strip()


def render_core(core: str, **values: str) -> str:
    # str.format would choke on other braces in the text; substitute known keys only.
    out = core
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out
