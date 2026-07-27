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

import os

import anthropic

from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase

SYSTEM_PROMPT = """\
You write PRE-VIEWING promotional material (spoiler-free) for a movie.
Whoever reads this hasn't seen it yet.

Rules:
- Don't reveal the ending, plot twists, who the killer/villain is, or any
  fact that only makes sense after seeing the movie.
- context_bullets and author_voice: max 3 each. Each one is a Claim with
  kind="fact" (verifiable, about production/cast/premise) or
  kind="interpretation" (your reading, doesn't need a source). If it's
  "fact", set source_id to null -- you have no real sources, this is a
  baseline with no retrieval, don't invent a fake source identifier.
- emotional_temperature: one sentence capturing the tone (a sensory
  metaphor).
- why_now: why watch it now, without spoiling anything.
- script: 1-3 blocks with timings in seconds, on-screen text, voiceover,
  and visual direction. The voiceover is also spoiler surface: the same
  rules apply there.

Call the emit_preshow_brief tool with the result. Don't write anything
outside the tool call.
"""

_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "kind": {"type": "string", "enum": ["fact", "interpretation"]},
        "source_id": {"type": ["string", "null"]},
    },
    "required": ["text", "kind", "source_id"],
}

_TOOL = {
    "name": "emit_preshow_brief",
    "description": "Emits the pre-viewing brief in the required format.",
    "input_schema": {
        "type": "object",
        "properties": {
            "context_bullets": {"type": "array", "maxItems": 3, "items": _CLAIM_SCHEMA},
            "author_voice": {"type": "array", "maxItems": 3, "items": _CLAIM_SCHEMA},
            "emotional_temperature": {"type": "string"},
            "why_now": {"type": "string"},
            "script": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_s": {"type": "integer"},
                        "end_s": {"type": "integer"},
                        "on_screen_text": {"type": "string"},
                        "voiceover": {"type": "string"},
                        "visual_direction": {"type": "string"},
                    },
                    "required": [
                        "start_s",
                        "end_s",
                        "on_screen_text",
                        "voiceover",
                        "visual_direction",
                    ],
                },
            },
        },
        "required": [
            "context_bullets",
            "author_voice",
            "emotional_temperature",
            "why_now",
            "script",
        ],
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
            tool_choice={"type": "tool", "name": "emit_preshow_brief"},
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
