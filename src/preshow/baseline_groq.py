"""Milestone 0 baseline generator, Groq variant.

Same non-negotiable as baseline.py: no retrieval, the model writes the
brief from parametric memory, typed output via forced tool use, measured
by the harness -- not promised in the text itself. The only thing that
changes is the provider, so this exists purely to make Milestone 0
runnable on Groq's free tier (no card, generous limits) instead of a paid
ANTHROPIC_API_KEY. See baseline_prompts.py for why the prompt/schema are
shared verbatim with the Anthropic version rather than duplicated.

Needs: pip install groq
       GROQ_API_KEY set as an env var, or a `GROQ_API_KEY=...` line in the
       repo-root .env (gitignored -- see src/preshow/env.py).
       Free key, no card: console.groq.com/keys
"""

from __future__ import annotations

import json

from groq import BadRequestError, Groq

from .baseline_prompts import (
    BRIEF_PROPERTIES,
    BRIEF_REQUIRED,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTION,
    TOOL_NAME,
)
from .env import read_env
from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase

_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": BRIEF_PROPERTIES,
            "required": BRIEF_REQUIRED,
        },
    },
}


class GroqBaselineGenerator:
    """Generator (Protocol) with no retrieval, same as
    AnthropicBaselineGenerator, running on Groq's free tier instead."""

    name = "baseline-groq"

    def __init__(self, client: Groq | None = None, model: str = "llama-3.3-70b-versatile"):
        self._client = client or Groq(api_key=read_env("GROQ_API_KEY"))
        self._model = model

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        data = self._call_with_retry(case)
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

    def _call_with_retry(self, case: TitleCase) -> dict:
        """Groq's tool-calling occasionally emits a call missing a
        required field (e.g. a script block skipping visual_direction)
        and the API rejects it with BadRequestError before this code ever
        sees a response -- observed live losing an entire run_eval.py run
        partway through (see docs/DESIGN.md D15's llm-judge section).
        Same corrective-retry pattern already used in
        webapp/research_assist.py's _call_llm: append what was wrong and
        ask again, instead of crashing the whole harness run on one bad
        generation. This changes retry BEHAVIOR only, not the shared
        SYSTEM_PROMPT/schema baseline_prompts.py deliberately keeps
        identical across providers (see that module's docstring) --
        Anthropic's tool-calling hasn't shown this failure mode, so it
        doesn't have this retry loop."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {case.title} ({case.year})"},
        ]
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=messages,
                    tools=[_TOOL],
                    tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                )
                call = resp.choices[0].message.tool_calls[0]
                return json.loads(call.function.arguments)
            except BadRequestError as e:
                if attempt == 2:
                    raise
                messages.append(
                    {
                        "role": "user",
                        "content": "That call was rejected: every object in an array must include "
                        "ALL of its required properties (context_bullets/author_voice items need "
                        "text, kind, AND source_id; script blocks need start_s, end_s, "
                        "on_screen_text, voiceover, AND visual_direction). Retry, filling in every "
                        "required field on every item.",
                    }
                )
        raise RuntimeError("unreachable")
