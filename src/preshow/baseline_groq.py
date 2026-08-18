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
from .groq_retry import GroqPacer, call_with_retry
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

    # This generator's prompt is the smallest of the three Groq-backed
    # callers (title only, no retrieved text), so it's the least likely
    # to hit a per-minute TPM wall -- but retrieval_groq.py/
    # research_assist.py both DID hit one live (D16 in docs/DESIGN.md)
    # and this generator previously had no RateLimitError handling at
    # all, just BadRequestError's corrective retry below. Now shares the
    # same pacing + backoff (groq_retry.py) as the other two.
    MIN_CALL_INTERVAL_S = 40.0

    # llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-18
    # (confirmed live: a call to it now 404s, "does not exist or you do
    # not have access to it") -- openai/gpt-oss-120b is Groq's own
    # documented replacement. All free/developer-tier models now share
    # identical limits regardless of size (RPM 30, RPD 1K, TPM 8K,
    # TPD 200K per Groq's rate-limits page, checked live) -- the old
    # "pick the smaller model for a bigger daily budget" lever from D16
    # no longer exists, every model gets the same quota now.
    def __init__(self, client: Groq | None = None, model: str = "openai/gpt-oss-120b"):
        self._client = client or Groq(api_key=read_env("GROQ_API_KEY"))
        self._model = model
        self._pacer = GroqPacer(self.MIN_CALL_INTERVAL_S)

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
        Corrective-retry: append what was wrong and ask again, instead of
        crashing the whole harness run on one bad generation. This
        changes retry BEHAVIOR only, not the shared SYSTEM_PROMPT/schema
        baseline_prompts.py deliberately keeps identical across providers
        (see that module's docstring) -- Anthropic's tool-calling hasn't
        shown this failure mode, so it doesn't have this retry loop.
        Pacing/RateLimitError retry now live in groq_retry.py, shared
        with retrieval_groq.py/research_assist.py (this generator
        previously had neither)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {case.title} ({case.year})"},
        ]

        def make_call() -> dict:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                messages=messages,
                tools=[_TOOL],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            )
            call = resp.choices[0].message.tool_calls[0]
            return json.loads(call.function.arguments)

        def correction_text(e: BadRequestError, attempt: int) -> str:
            return (
                "That call was rejected: every object in an array must include "
                "ALL of its required properties (context_bullets/author_voice items need "
                "text, kind, AND source_id; script blocks need start_s, end_s, "
                "on_screen_text, voiceover, AND visual_direction). Retry, filling in every "
                "required field on every item."
            )

        # max_attempts=3 -- kept at its original value, not raised to
        # match the other two generators' 5 (see D16's note on
        # retrieval_groq.py for why 5 mattered there specifically).
        return call_with_retry(make_call, messages, correction_text, self._pacer, max_attempts=3)
