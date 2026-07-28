"""Milestone 0 baseline generator.

A single call to the Anthropic API, WITHOUT retrieval: the model writes the
brief from its own parametric memory. This is the corrected version of the
original prompt (see section 2 of CLAUDE.md): same lack of retrieval, but
typed output via tool use instead of markdown with a self-labeled traffic
light, measured by the harness instead of promised in the generated text
itself.

It isn't meant to be good. It's here to measure the real starting point --
if the leakage_rate comes out high on long-tail titles, that's the
quantitative justification for Milestone 1 (retrieval), not a design
assumption.
"""

from __future__ import annotations

import anthropic

from .baseline_prompts import (
    BRIEF_PROPERTIES,
    BRIEF_REQUIRED,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTION,
    TOOL_NAME,
)
from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase

_TOOL = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": BRIEF_PROPERTIES,
        "required": BRIEF_REQUIRED,
    },
}


class AnthropicBaselineGenerator:
    """Generator (Protocol) with no retrieval. `corpus` is ignored on
    purpose: that's exactly the point this baseline exists to measure."""

    name = "baseline"

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = "claude-sonnet-5"):
        self._client = client or anthropic.Anthropic()
        self._model = model

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[
                {
                    "role": "user",
                    "content": f"Title: {case.title} ({case.year})",
                }
            ],
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        data = block.input
        return PreShowBrief(
            title_id=case.title_id,
            context_bullets=[Claim(**c) for c in data["context_bullets"]],
            author_voice=[Claim(**c) for c in data["author_voice"]],
            emotional_temperature=data["emotional_temperature"],
            why_now=data["why_now"],
            script=[ScriptBlock(**b) for b in data["script"]],
        )

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive:
        raise NotImplementedError("Milestone 0 only covers pre_show; deep_dive lands in a later milestone")
