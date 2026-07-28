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

from groq import Groq

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
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Title: {case.title} ({case.year})"},
            ],
            tools=[_TOOL],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        call = resp.choices[0].message.tool_calls[0]
        data = json.loads(call.function.arguments)
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
